@"
# DO NOT USE — Historical Reference Only

This folder contains scripts from earlier project iterations. They are
preserved for provenance (e.g., to reproduce the F1=0.2254 DictaBERT
result reported in the status doc) but are **not part of the active
pipeline**.

## Rules

- Nothing in this folder should be called by any active script, sbatch,
  or README.
- Do not modify files in here. If a fix is needed, it belongs in a
  current location (``scripts/`` or ``scripts/slurm/<model>/``).
- If you need behavior that lives here, **copy** the file to its proper
  location and update it there.

## Why it's preserved

- ``train_*.sbatch`` — early flat structure, replaced by per-model
  ablation folders (``scripts/slurm/alephbert/``,
  ``scripts/slurm/dictabert/``, etc.) with ``P0_baseline``,
  ``P1_add_weights``, ``P2_add_focal`` conventions per the MASTER_PLAN v7.
- ``evaluate_teacher.py`` — older version, superseded by
  ``scripts/evaluate_teacher.py``.
- ``verify_data_integrity.py`` — older version, superseded by
  ``scripts/verify_data_integrity.py``.
- ``train_twostep_*.sbatch`` — two-step pipeline variants from v6, the
  final paper uses single-step token classification.
- ``setup_cluster.sh``, ``run_experiments.sh`` — early orchestration
  scripts, replaced by the per-model sbatch submissions.

## Audit

Run this from ``recipe_project/`` root to confirm nothing outside
``old/`` references these files:

``````powershell
Select-String -Path "**\*.sh","**\*.sbatch","**\*.py","**\*.md" ``
    -Pattern "slurm/old|slurm\\old" -SimpleMatch 2>``$null |
    Where-Object { ``$_.Path -notmatch "slurm[\\/]old[\\/]" }
``````

Expected output: empty (no active file references ``old/``).
"@ | Out-File -FilePath scripts/slurm/old/README.md -Encoding ascii

# Verify
Get-Content scripts/slurm/old/README.md | Select-Object -First 5