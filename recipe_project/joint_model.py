"""
Joint Intent + Slot Model with CRF for Hebrew Recipe Modification Extraction.

Architecture:
  BERT encoder
    → [CLS] → intent classifier (has_modification yes/no)
    → intent class → nn.Embedding → broadcast to all tokens
    → [token_enc ‖ intent_embed] → Linear → emission scores
    → CRF → valid BIO sequence

Key fixes:
  - Bug #1: Discrete intent embedding (nn.Embedding), NOT raw CLS vector
  - CRF enforces valid BIO transitions (no orphan I tags)
  - Scheduled sampling: gold intent → predicted intent over epochs
  - Handles -100 labels (special tokens) by masking in CRF

Usage:
    from src.models.joint_model import JointModelWithCRF
    model = JointModelWithCRF("dicta-il/dictabert")
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchcrf import CRF


class JointModelWithCRF(nn.Module):
    """Joint intent classification + BIO slot filling with CRF."""

    def __init__(
        self,
        model_name="dicta-il/dictabert",
        num_intent_labels=2,
        num_slot_labels=9,
        dropout_rate=0.1,
        intent_embed_dim=None,
    ):
        """
        Args:
            model_name: HuggingFace model ID for pretrained Hebrew BERT
            num_intent_labels: 2 (has_modification: yes/no)
            num_slot_labels: 9 (O + B/I for 4 aspects)
            dropout_rate: Dropout probability
            intent_embed_dim: Size of discrete intent embedding (default: hidden//4)
        """
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # ── Intent head ──────────────────────────────────────────────
        self.intent_classifier = nn.Linear(hidden, num_intent_labels)

        # ── Discrete intent embedding (Bug #1 fix) ──────────────────
        # Embeds the predicted/gold intent class (0 or 1) into a vector
        # that gets concatenated to every token embedding.
        # OLD (wrong): used raw CLS vector → leaked continuous info
        # NEW (fixed): nn.Embedding of discrete class → clean signal
        if intent_embed_dim is None:
            intent_embed_dim = hidden // 4
        self.intent_embed_dim = intent_embed_dim
        self.intent_embedding = nn.Embedding(num_intent_labels, intent_embed_dim)

        # ── Slot emission (for CRF) ─────────────────────────────────
        # Input = [token_encoding (hidden) ‖ intent_embedding (intent_embed_dim)]
        self.slot_emission = nn.Linear(hidden + intent_embed_dim, num_slot_labels)

        # ── CRF layer ────────────────────────────────────────────────
        # Learns transition scores between BIO labels.
        # Prevents invalid sequences like O → I-SUB or B-SUB → I-QTY.
        self.crf = CRF(num_slot_labels, batch_first=True)

    @staticmethod
    def _gold_probability(epoch):
        """Scheduled sampling: reduce gold intent usage over epochs."""
        if epoch < 3:
            return 1.0    # epochs 0-2: 100% gold
        elif epoch < 7:
            return 0.7    # epochs 3-6: 70% gold
        else:
            return 0.5    # epochs 7+:  50% gold

    def forward(
        self,
        input_ids,
        attention_mask,
        labels_intent=None,
        labels_slot=None,
        epoch=None,
        training=True,
    ):
        """
        Forward pass.

        Args:
            input_ids:      [batch, seq_len]
            attention_mask:  [batch, seq_len]
            labels_intent:   [batch] — 0 or 1 (optional, for training)
            labels_slot:     [batch, seq_len] — BIO label IDs, -100 for ignore
            epoch:           current epoch (for scheduled sampling)
            training:        whether in training mode

        Returns:
            If labels_slot is provided (training):
                intent_logits: [batch, 2]
                slot_loss:     scalar (CRF negative log-likelihood)
            If labels_slot is None (inference):
                intent_logits: [batch, 2]
                slot_preds:    list of lists (Viterbi-decoded BIO sequences)
        """
        # ── BERT encoding ────────────────────────────────────────────
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]    # [batch, hidden]
        token_emb = outputs.last_hidden_state            # [batch, seq, hidden]

        # ── Intent classification ────────────────────────────────────
        intent_logits = self.intent_classifier(cls_emb)  # [batch, 2]

        # ── Get intent class for conditioning ────────────────────────
        if training and labels_intent is not None and epoch is not None:
            gold_prob = self._gold_probability(epoch)
            if random.random() < gold_prob:
                intent_class = labels_intent
            else:
                intent_class = torch.argmax(intent_logits, dim=-1)
        else:
            intent_class = torch.argmax(intent_logits, dim=-1)

        # ── Intent embedding → broadcast to all tokens ───────────────
        intent_emb = self.intent_embedding(intent_class)          # [batch, D]
        intent_exp = intent_emb.unsqueeze(1).expand(
            -1, token_emb.size(1), -1
        )                                                          # [batch, seq, D]

        # ── Emission scores ──────────────────────────────────────────
        combined = torch.cat([token_emb, intent_exp], dim=-1)     # [batch, seq, H+D]
        emissions = self.slot_emission(self.dropout(combined))     # [batch, seq, 9]

        # ── CRF: loss (train) or decode (inference) ──────────────────
        if labels_slot is not None:
            # CRF cannot handle -100 → replace with 0, mask those positions out
            clean_labels = labels_slot.clone()
            ignore_mask = (labels_slot == -100)
            clean_labels[ignore_mask] = 0

            crf_mask = attention_mask.bool() & (~ignore_mask)

            slot_loss = -self.crf(
                emissions, clean_labels, mask=crf_mask, reduction="mean"
            )
            return intent_logits, slot_loss
        else:
            crf_mask = attention_mask.bool()
            slot_preds = self.crf.decode(emissions, mask=crf_mask)
            return intent_logits, slot_preds
