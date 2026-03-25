"""
Joint Model: BERT + CRF for Token Classification
===================================================
Architecture: BERT encoder → linear projection → CRF layer

This is the simplified V7 architecture that DROPS intent conditioning.
The V6 version had an intent head with scheduled sampling, but the
V7 critique recommends removing it because:
  - Cascade failure risk (wrong intent → wrong spans)
  - Expected gain is only +1-2 F1 points
  - Adds architectural complexity that is hard to debug
  - Makes the CRF ablation (A4) less clean

The CRF layer enforces valid BIO transitions:
  - No orphan I-X tags (I-X must follow B-X or I-X)
  - Valid B→I→I sequences within each aspect
  - Viterbi decoding at inference for globally optimal tag sequence

Usage:
    from src.models.joint_model import BertCRFModel

    model = BertCRFModel(
        model_name="dicta-il/dictabert",
        num_labels=9,   # BIO: O + 4 aspects × (B, I)
        dropout_rate=0.1,
    )

    # Training: returns CRF NLL loss (scalar)
    loss = model(input_ids, attention_mask, labels=labels)

    # Inference: returns list of predicted tag sequences
    predictions = model(input_ids, attention_mask)  # no labels → inference mode
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class BertCRFModel(nn.Module):
    """BERT encoder + Linear + CRF for BIO sequence tagging.

    The CRF layer learns transition probabilities between BIO tags,
    preventing invalid sequences like O → I-SUBSTITUTION. This is
    especially important for Hebrew text where tokenization can split
    morphologically rich words into subwords.

    Args:
        model_name: HuggingFace model ID (e.g., 'dicta-il/dictabert')
        num_labels: Number of BIO labels (default: 9)
            O, B-SUBSTITUTION, I-SUBSTITUTION, B-QUANTITY, I-QUANTITY,
            B-TECHNIQUE, I-TECHNIQUE, B-ADDITION, I-ADDITION
        dropout_rate: Dropout probability on encoder output (default: 0.1)
        class_weights: Optional tensor of shape (num_labels,) for weighting
            emission scores. If provided, emission scores for each label
            are multiplied by the corresponding weight before CRF.
    """

    def __init__(
        self,
        model_name: str = "dicta-il/dictabert",
        num_labels: int = 9,
        dropout_rate: float = 0.1,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()

        self.num_labels = num_labels

        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # typically 768

        # Projection: hidden → num_labels (emission scores for CRF)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # CRF layer (learns transition matrix between BIO tags)
        self.crf = CRF(num_labels, batch_first=True)

        # Optional class weights for emission scaling
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run BERT + linear to get emission scores.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            emissions: (batch, seq_len, num_labels)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        # Apply class weights to emissions if provided
        if self.class_weights is not None:
            emissions = emissions * self.class_weights.unsqueeze(0).unsqueeze(0)

        return emissions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """Forward pass.

        Training mode (labels provided):
            Returns the negative log-likelihood loss from CRF.

        Inference mode (no labels):
            Returns list of predicted tag sequences via Viterbi decoding.

        Args:
            input_ids: (batch, seq_len) — tokenized input
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
            labels: (batch, seq_len) — BIO label IDs, -100 for special tokens.
                If None, runs inference (Viterbi decoding).

        Returns:
            If labels provided: scalar loss (CRF negative log-likelihood)
            If labels is None: list of lists of predicted label IDs
        """
        emissions = self._get_emissions(input_ids, attention_mask)

        if labels is not None:
            # ── Training: compute CRF NLL loss ───────────────────────
            # CRF cannot handle -100 (ignore index). We need to:
            # 1. Create a clean label tensor replacing -100 with 0 (O tag)
            # 2. Create a CRF mask that excludes -100 positions
            crf_labels = labels.clone()
            crf_mask = (labels != -100) & (attention_mask == 1)

            # CRITICAL FIX: pytorch-crf requires mask[:, 0] == True for every
            # sequence in the batch. [CLS] always has label=-100 which sets
            # crf_mask[:, 0] = False → "mask of the first timestep must all be on"
            # Force it True and set the label to O (0). The [CLS] token's
            # contribution to the CRF path is harmless since it always predicts O.
            crf_mask[:, 0] = True

            # Replace -100 with 0 (O tag) — CRF needs valid indices everywhere
            crf_labels[crf_labels == -100] = 0

            log_likelihood = self.crf(
                emissions,
                crf_labels,
                mask=crf_mask.bool(),
                reduction="mean",
            )
            loss = -log_likelihood  # negate: CRF returns log-prob, we want NLL
            return loss

        else:
            # ── Inference: Viterbi decoding ──────────────────────────
            crf_mask = attention_mask.bool()
            best_paths = self.crf.decode(emissions, mask=crf_mask)
            return best_paths  # list of lists of label IDs


# =============================================================================
# Backward-compatible alias for existing import statements
# =============================================================================
# If your train_joint.py imports JointModelWithCRF, this alias ensures
# it still works after the simplification.
JointModelWithCRF = BertCRFModel