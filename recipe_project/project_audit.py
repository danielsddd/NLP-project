#!/usr/bin/env python3
"""
=============================================================================
PROJECT AUDIT SCRIPT — Hebrew Recipe Modification Extraction (NLP 2025a)
=============================================================================
Author : Daniel Simanovsky (adapted from MASTER_PLAN_v8.1)
Purpose: Map every file on the cluster, cross-reference HuggingFace and
         GitHub, surface version conflicts, identify debris, and produce
         a prioritised action plan so nothing is lost before submission.

Usage (on cluster or locally):
    python project_audit.py                           # full audit
    python project_audit.py --root /path/to/project  # custom root
    python project_audit.py --hf-token hf_xxxx       # explicit HF token
    python project_audit.py --no-hf                  # skip HF checks
    python project_audit.py --no-git                 # skip git checks
    python project_audit.py --report-md audit.md     # save markdown report

Outputs:
    • Coloured terminal summary  (rich)
    • Markdown report            (--report-md, default: audit_report.md)
    • JSON snapshot              (audit_snapshot.json, machine-readable)
=============================================================================
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── optional rich for terminal colouring ──────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None  # type: ignore
    def rprint(*a, **kw): print(*a)   # type: ignore

# ── optional HuggingFace Hub ───────────────────────────────────────────────
try:
    from huggingface_hub import HfApi, list_models, list_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ── optional GitPython ────────────────────────────────────────────────────
try:
    import git as gitpython
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


# =============================================================================
# CONFIGURATION — edit these to match your project
# =============================================================================

HF_USERNAME = "DanielDDDS"

# Canonical src/ files from §10.1 — must exist in repo
CANONICAL_SRC = [
    "src/teacher_labeling/generate_labels.py",
    "src/preprocessing/__init__.py",
    "src/preprocessing/prepare_data.py",
    "src/preprocessing/prepare_data_merged.py",
    "src/models/train_student.py",
    "src/models/train_joint.py",
    "src/models/joint_model.py",
    "src/evaluation/evaluate.py",
    "src/baselines/run_baselines.py",
    "src/utils/__init__.py",
    "src/utils/class_weights.py",
]

# Canonical scripts from §10.1 — must exist in repo
CANONICAL_SCRIPTS = [
    "scripts/evaluate_teacher.py",
    "scripts/verify_data_integrity.py",
    "scripts/download_silver_labels.sh",
    "scripts/run_all_ablations.sh",
    "scripts/local/prepare_gold_for_eval.py",
    "scripts/local/compute_kappa.py",
    "scripts/local/apply_fix5.ps1",  # archival, not critical
]

# SLURM model slugs expected from §5.3
SLURM_MODEL_SLUGS = [
    "alephbert",
    "dictabert",
    "dictabert_crf",
    "dictabert_large",
    "hebert",
    "mbert",
    "xlmr",
]

# Expected sbatch suffixes per model (P0-P5 progressive + ablations + evaluate_all)
# §5.3 says 23 files per model
EXPECTED_SBATCH_SUFFIXES = [
    "P0_preprocess.sbatch",
    "P1_train.sbatch",
    "P2_train_classweights.sbatch",
    "P3_train_focalloss.sbatch",
    "P4_train_crf.sbatch",
    "P5_train_full.sbatch",
    "evaluate_all.sbatch",
    "A1a_no_weights.sbatch",
    "A1b_uniform_weights.sbatch",
    "A1c_weights_only.sbatch",
    "A2_downsample_1.sbatch",
    "A2_downsample_2.sbatch",
    "A2_downsample_4.sbatch",
    "A5_data_25pct.sbatch",
    "A5_data_50pct.sbatch",
    "A5_data_75pct.sbatch",
    "A6_no_enriched.sbatch",
    "A7_io_scheme.sbatch",
    "A8_unanimous.sbatch",
]

# Data files that must exist (§10.2)
REQUIRED_DATA_FILES = [
    "data/raw_youtube/threads.jsonl",
    "data/raw_youtube/threads_positives_focus.jsonl",
    "data/gold_validation/gold_final.jsonl",
]

# Data files that are the v2 targets (should exist if Track A ran)
V2_DATA_FILES = [
    "data/silver_labels/teacher_output_v2.jsonl",
    "data/silver_labels/threads_positives_focus_labeled_v2.jsonl",
    "data/silver_labels/pilot_output.jsonl",
    "data/silver_labels/CHANGELOG.md",
    "data/processed_v2/train_merged.jsonl",
    "data/processed_v2/val.jsonl",
    "data/processed_v2/test.jsonl",
    "results/teacher_upper_bound_v2.json",
]

# Debris patterns — files matching these should be reviewed for deletion
DEBRIS_PATTERNS = [
    "MASTER_PLAN_v7.txt",
    "MASTER_PLAN_v7_addendum.txt",
    "MASTER_PLAN_v8.txt",
    "MASTER_PLAN_v8.1_patch.txt",
    "experiment_scripts_v2",
    "fix_*.sh",
    "Pipfile",
    "Pipfile.lock",
    "*.bak",
    "*.bak.*",
    "*.old",
    "*_backup*",
    "*_copy*",
    "*_orig*",
    "*_BACKUP*",
    "*_OLD*",
    "*_temp*",
    "*_tmp*",
    "*.swp",
    ".DS_Store",
    "__pycache__",
]

# Folders to skip entirely when scanning
SKIP_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    "node_modules", ".venv", "venv", "env", ".conda",
}

# Patterns that suggest a file is a "newer version supersedes older"
VERSION_PATTERNS = [
    (r"(.+?)_v(\d+)(\..+)?$",        "versioned"),      # foo_v2.jsonl
    (r"(.+?)_V(\d+)(\..+)?$",        "versioned_cap"),  # foo_V2.jsonl
    (r"(.+?)_gold_final_v(\d+)",      "gold_version"),
    (r"(.+?)_final(.+)?$",            "final"),
    (r"(.+?)_final_v(\d+)",           "final_versioned"),
    (r"MASTER_PLAN_v(\d+)",           "plan_version"),
]

# HF datasets that are known/expected
KNOWN_HF_DATASETS = [
    "recipe-modifications",
]

# What should be on GitHub (must be committed, not slurm-only)
MUST_BE_ON_GIT = (
    CANONICAL_SRC
    + CANONICAL_SCRIPTS
    + [
        "requirements.txt",
        "README.md",
        "MASTER_PLAN_v8_1.txt",
    ]
)


# =============================================================================
# HELPERS
# =============================================================================

def sha256_file(path: Path, chunk=65536) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                data = f.read(chunk)
                if not data:
                    break
                h.update(data)
        return h.hexdigest()[:12]
    except Exception:
        return "unreadable"


def file_age_days(path: Path) -> float:
    try:
        mtime = path.stat().st_mtime
        return (datetime.now().timestamp() - mtime) / 86400
    except Exception:
        return -1.0


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


def run_cmd(cmd: str) -> Tuple[int, str, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def matches_debris(name: str) -> bool:
    for pat in DEBRIS_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def detect_version_group(name: str):
    """Return (base_name, version_int) if name looks versioned, else None."""
    for pat, _ in VERSION_PATTERNS:
        m = re.match(pat, name, re.IGNORECASE)
        if m:
            groups = m.groups()
            for g in groups:
                if g and g.isdigit():
                    return (groups[0], int(g))
    return None


# =============================================================================
# SCANNER
# =============================================================================

def scan_project(root: Path) -> Dict[str, Any]:
    """Walk the project root and classify every file."""
    result = {
        "canonical_src": {},       # path -> {exists, size, age_days, sha}
        "canonical_scripts": {},
        "slurm_map": {},           # model -> {found: [...], missing: [...]}
        "data_required": {},
        "data_v2": {},
        "debris": [],              # list of {path, reason}
        "version_groups": {},      # base_name -> [list of versioned files]
        "results_files": [],
        "all_files_count": 0,
        "all_files_size": 0,
        "untracked_interesting": [],
    }

    # ── Canonical src and scripts ─────────────────────────────────────────
    for rel in CANONICAL_SRC:
        p = root / rel
        result["canonical_src"][rel] = {
            "exists": p.exists(),
            "size": p.stat().st_size if p.exists() else 0,
            "age_days": file_age_days(p) if p.exists() else -1,
            "sha": sha256_file(p) if p.exists() else "",
        }

    for rel in CANONICAL_SCRIPTS:
        p = root / rel
        result["canonical_scripts"][rel] = {
            "exists": p.exists(),
            "size": p.stat().st_size if p.exists() else 0,
            "age_days": file_age_days(p) if p.exists() else -1,
            "sha": sha256_file(p) if p.exists() else "",
        }

    # ── Required data files ───────────────────────────────────────────────
    for rel in REQUIRED_DATA_FILES:
        p = root / rel
        result["data_required"][rel] = {
            "exists": p.exists(),
            "size": p.stat().st_size if p.exists() else 0,
        }

    # ── v2 data files ─────────────────────────────────────────────────────
    for rel in V2_DATA_FILES:
        p = root / rel
        result["data_v2"][rel] = {
            "exists": p.exists(),
            "size": p.stat().st_size if p.exists() else 0,
        }

    # ── SLURM sbatch tree ─────────────────────────────────────────────────
    for slug in SLURM_MODEL_SLUGS:
        slurm_dir = root / "scripts" / "slurm" / slug
        found, missing = [], []
        for suf in EXPECTED_SBATCH_SUFFIXES:
            fp = slurm_dir / suf
            if fp.exists():
                found.append(suf)
            else:
                missing.append(suf)
        # also pick up any unexpected sbatch files
        unexpected = []
        if slurm_dir.exists():
            for f in sorted(slurm_dir.iterdir()):
                if f.is_file() and f.name.endswith(".sbatch"):
                    if f.name not in EXPECTED_SBATCH_SUFFIXES:
                        unexpected.append(f.name)
        result["slurm_map"][slug] = {
            "dir_exists": slurm_dir.exists(),
            "found": found,
            "missing": missing,
            "unexpected": unexpected,
            "completeness_pct": round(
                100 * len(found) / len(EXPECTED_SBATCH_SUFFIXES), 1
            ) if EXPECTED_SBATCH_SUFFIXES else 0,
        }

    # ── Full walk: debris + version groups + results + sizes ──────────────
    version_registry: Dict[str, List[str]] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # prune skipped dirs in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        dp = Path(dirpath)
        rel_dir = dp.relative_to(root)

        for fname in filenames:
            fpath = dp / fname
            try:
                fsize = fpath.stat().st_size
            except Exception:
                fsize = 0

            result["all_files_count"] += 1
            result["all_files_size"] += fsize

            rel_path = str(fpath.relative_to(root))

            # debris detection
            if matches_debris(fname) or matches_debris(str(rel_dir)):
                result["debris"].append({
                    "path": rel_path,
                    "size": fsize,
                    "age_days": round(file_age_days(fpath), 1),
                    "reason": "matches debris pattern",
                })

            # version group detection
            stem = Path(fname).stem
            vg = detect_version_group(stem)
            if vg:
                base, ver = vg
                key = f"{rel_dir}/{base}" if str(rel_dir) != "." else base
                if key not in version_registry:
                    version_registry[key] = []
                version_registry[key].append({
                    "path": rel_path,
                    "version": ver,
                    "size": fsize,
                    "age_days": round(file_age_days(fpath), 1),
                })

            # results files detection
            if "results" in rel_path and fname.endswith(".json"):
                result["results_files"].append({
                    "path": rel_path,
                    "size": fsize,
                    "age_days": round(file_age_days(fpath), 1),
                })

    # keep only groups with >1 member (actually versioned sets)
    result["version_groups"] = {
        k: sorted(v, key=lambda x: x["version"])
        for k, v in version_registry.items()
        if len(v) > 1
    }

    return result


# =============================================================================
# GIT AUDIT
# =============================================================================

def audit_git(root: Path) -> Dict[str, Any]:
    """Check git status, untracked files, last commit, remote URLs."""
    info: Dict[str, Any] = {
        "available": False,
        "remote_url": "",
        "current_branch": "",
        "last_commit": "",
        "last_commit_date": "",
        "untracked": [],
        "modified": [],
        "must_be_committed_status": {},
        "ahead_behind": "",
    }

    git_dir = root / ".git"
    if not git_dir.exists():
        info["note"] = "No .git directory found in project root"
        return info

    info["available"] = True

    rc, out, _ = run_cmd(f"git -C '{root}' remote get-url origin 2>/dev/null")
    info["remote_url"] = out

    rc, out, _ = run_cmd(f"git -C '{root}' rev-parse --abbrev-ref HEAD")
    info["current_branch"] = out

    rc, out, _ = run_cmd(f"git -C '{root}' log -1 --format='%h %s (%ar)'")
    info["last_commit"] = out

    rc, out, _ = run_cmd(f"git -C '{root}' log -1 --format='%ci'")
    info["last_commit_date"] = out

    rc, out, _ = run_cmd(f"git -C '{root}' status --porcelain")
    for line in out.splitlines():
        if line.startswith("??"):
            info["untracked"].append(line[3:].strip())
        elif line[:2].strip():
            info["modified"].append(line[3:].strip())

    rc, out, _ = run_cmd(
        f"git -C '{root}' rev-list --count --left-right HEAD...@{{upstream}} 2>/dev/null"
    )
    if rc == 0 and "\t" in out:
        ahead, behind = out.split("\t")
        info["ahead_behind"] = f"↑{ahead} ahead, ↓{behind} behind remote"

    for rel in MUST_BE_ON_GIT:
        rc, out, _ = run_cmd(
            f"git -C '{root}' ls-files --error-unmatch '{rel}' 2>/dev/null"
        )
        info["must_be_committed_status"][rel] = "tracked" if rc == 0 else "UNTRACKED/MISSING"

    return info


# =============================================================================
# HUGGINGFACE AUDIT
# =============================================================================

def audit_huggingface(hf_token: Optional[str]) -> Dict[str, Any]:
    """List all models and datasets for HF_USERNAME, detect version pairs."""
    info: Dict[str, Any] = {
        "available": HF_AVAILABLE,
        "username": HF_USERNAME,
        "datasets": [],
        "models": [],
        "version_pairs": [],
        "errors": [],
    }

    if not HF_AVAILABLE:
        info["errors"].append("huggingface_hub not installed — run: pip install huggingface_hub")
        return info

    try:
        api = HfApi(token=hf_token)

        # Datasets
        ds_list = list(api.list_datasets(author=HF_USERNAME))
        for ds in ds_list:
            card = {}
            try:
                card = api.dataset_info(ds.id).card_data or {}
            except Exception:
                pass
            tags = getattr(ds, "tags", []) or []
            info["datasets"].append({
                "id": ds.id,
                "private": getattr(ds, "private", False),
                "last_modified": str(getattr(ds, "last_modified", "")),
                "downloads": getattr(ds, "downloads", 0),
                "tags": tags,
            })

        # Models
        model_list = list(api.list_models(author=HF_USERNAME))
        for m in model_list:
            info["models"].append({
                "id": m.id,
                "private": getattr(m, "private", False),
                "last_modified": str(getattr(m, "last_modified", "")),
                "downloads": getattr(m, "downloads", 0),
                "pipeline_tag": getattr(m, "pipeline_tag", ""),
            })

        # Detect version pairs across datasets + models
        all_ids = [x["id"] for x in info["datasets"]] + [x["id"] for x in info["models"]]
        seen_bases: Dict[str, List[str]] = {}
        for hf_id in all_ids:
            name = hf_id.split("/")[-1]
            vg = detect_version_group(name)
            if vg:
                base, ver = vg
                seen_bases.setdefault(base, []).append(hf_id)
            # also check for "final" vs "final_v2" patterns
            for keyword in ["final", "gold", "v1", "v2", "v3"]:
                if keyword in name.lower():
                    generic = re.sub(r"[_\-]?(v\d+|final|gold)[_\-]?", "", name, flags=re.I).strip("_-")
                    seen_bases.setdefault(f"hf_{generic}", []).append(hf_id)
                    break

        for base, ids in seen_bases.items():
            unique = list(dict.fromkeys(ids))
            if len(unique) > 1:
                info["version_pairs"].append({"base": base, "versions": unique})

    except Exception as exc:
        info["errors"].append(str(exc))

    return info


# =============================================================================
# CROSS-REFERENCE: SLURM ↔ HF ↔ GIT
# =============================================================================

def cross_reference(scan: Dict, git: Dict, hf: Dict) -> List[Dict]:
    """
    Produce a list of ACTION items with priority levels:
      P0 = blocker (data/code loss risk)
      P1 = important (should fix before submission)
      P2 = cleanup (nice-to-have)
    """
    actions: List[Dict] = []

    def add(priority: str, category: str, item: str, detail: str, suggestion: str):
        actions.append({
            "priority": priority,
            "category": category,
            "item": item,
            "detail": detail,
            "suggestion": suggestion,
        })

    # ── Missing canonical files ───────────────────────────────────────────
    for rel, info in scan["canonical_src"].items():
        if not info["exists"]:
            add("P0", "missing_canonical",
                rel,
                "File listed in §10.1 does not exist on disk",
                f"Create or restore: {rel}")

    for rel, info in scan["canonical_scripts"].items():
        if not info["exists"]:
            sev = "P1" if "apply_fix5" not in rel else "P2"
            add(sev, "missing_canonical",
                rel,
                "Script listed in §10.1 does not exist on disk",
                f"Create or restore: {rel}")

    # ── Missing required data ─────────────────────────────────────────────
    for rel, info in scan["data_required"].items():
        if not info["exists"]:
            add("P0", "missing_data",
                rel,
                "Required data file is absent — pipeline cannot run",
                f"Download or restore from HF/backup: {rel}")

    # ── SLURM completeness ───────────────────────────────────────────────
    for slug, sm in scan["slurm_map"].items():
        if not sm["dir_exists"]:
            add("P1", "slurm_missing_dir",
                f"scripts/slurm/{slug}/",
                "Entire sbatch folder for this model is absent",
                f"Create directory and add sbatch files for model '{slug}'")
        elif sm["missing"]:
            add("P1", "slurm_incomplete",
                f"scripts/slurm/{slug}/",
                f"{len(sm['missing'])} sbatch file(s) missing: {', '.join(sm['missing'][:5])}{'...' if len(sm['missing'])>5 else ''}",
                "Generate missing sbatch files following §5.3 template")

    # ── Git tracking ──────────────────────────────────────────────────────
    if git.get("available"):
        for rel, status in git.get("must_be_committed_status", {}).items():
            if status != "tracked":
                add("P0", "git_untracked",
                    rel,
                    "This file must be on GitHub but is NOT tracked by git",
                    f"git add {rel} && git commit -m 'feat: add {Path(rel).name}'")

        if git.get("modified"):
            for mf in git["modified"]:
                add("P1", "git_dirty",
                    mf,
                    "File has uncommitted changes",
                    f"git add {mf} && git commit -m 'update: {Path(mf).name}'")

        if git.get("ahead_behind") and "↑0" not in git["ahead_behind"]:
            add("P1", "git_unpushed",
                "origin/main",
                f"Local commits not pushed: {git['ahead_behind']}",
                "git push origin main")

    # ── HuggingFace: check known datasets exist ───────────────────────────
    hf_ds_names = [x["id"].split("/")[-1] for x in hf.get("datasets", [])]
    for expected in KNOWN_HF_DATASETS:
        if expected not in hf_ds_names:
            add("P0", "hf_missing_dataset",
                f"{HF_USERNAME}/{expected}",
                "Expected HF dataset not found under your account",
                f"Upload dataset: huggingface-cli upload {expected}")

    # ── HuggingFace: version pairs (old should be archived/deleted) ───────
    for vp in hf.get("version_pairs", []):
        ids = vp["versions"]
        add("P2", "hf_version_conflict",
            " vs ".join(ids),
            f"Multiple versions detected on HuggingFace: {ids}",
            "Keep the latest (V2/gold_final_V2). Archive or delete older version. "
            "Add a README.md to the kept repo clarifying it supersedes the old one.")

    # ── HuggingFace: private repos that probably should be public ─────────
    for ds in hf.get("datasets", []):
        if ds["private"]:
            add("P1", "hf_private_dataset",
                ds["id"],
                "Dataset is PRIVATE on HuggingFace — reviewers cannot access it",
                "Make public: api.update_repo_visibility(repo_id, private=False, repo_type='dataset')")

    for m in hf.get("models", []):
        if m["private"]:
            add("P1", "hf_private_model",
                m["id"],
                "Model is PRIVATE — reviewers cannot access checkpoint",
                "Make public or add reviewer as collaborator")

    # ── Debris / backup files ─────────────────────────────────────────────
    for d in scan["debris"]:
        add("P2", "debris",
            d["path"],
            f"Matches debris pattern ({d['size']} bytes, {d['age_days']}d old)",
            "Review and delete, or move to docs/archive/")

    # ── Version conflicts on disk ─────────────────────────────────────────
    for base, versions in scan["version_groups"].items():
        newest = max(versions, key=lambda x: x["version"])
        older = [v for v in versions if v["version"] != newest["version"]]
        for old in older:
            add("P2", "disk_version_conflict",
                old["path"],
                f"Superseded by {newest['path']} (v{newest['version']} > v{old['version']})",
                "Delete or move to docs/archive/ after verifying the newer version is stable")

    # ── v2 data not yet present (Track A not done) ───────────────────────
    v2_exists = sum(1 for v in scan["data_v2"].values() if v["exists"])
    v2_total = len(scan["data_v2"])
    if v2_exists == 0:
        add("P1", "track_a_not_started",
            "data/silver_labels/teacher_output_v2.jsonl",
            "No v2 data files found — Track A (re-labeling) has not run yet",
            "Execute §4 of MASTER_PLAN_v8.1 to generate v2 labels")
    elif v2_exists < v2_total:
        add("P1", "track_a_incomplete",
            "data/processed_v2/",
            f"Only {v2_exists}/{v2_total} v2 data files exist — Track A may be partially complete",
            "Check §4.6 and continue pipeline to completion")

    # Sort by priority
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    actions.sort(key=lambda x: priority_order.get(x["priority"], 9))
    return actions


# =============================================================================
# REPORT GENERATORS
# =============================================================================

PRIORITY_EMOJI = {"P0": "🔴", "P1": "🟡", "P2": "🟢"}
CATEGORY_LABELS = {
    "missing_canonical":     "Missing canonical file",
    "missing_data":          "Missing data file",
    "slurm_missing_dir":     "SLURM dir absent",
    "slurm_incomplete":      "SLURM incomplete",
    "git_untracked":         "NOT tracked by Git",
    "git_dirty":             "Git dirty (uncommitted)",
    "git_unpushed":          "Git unpushed commits",
    "hf_missing_dataset":    "HF dataset missing",
    "hf_version_conflict":   "HF version conflict",
    "hf_private_dataset":    "HF dataset private",
    "hf_private_model":      "HF model private",
    "debris":                "Debris / backup file",
    "disk_version_conflict": "Old version on disk",
    "track_a_not_started":   "Track A not started",
    "track_a_incomplete":    "Track A incomplete",
}


def print_terminal_report(scan: Dict, git: Dict, hf: Dict, actions: List[Dict]):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("\n" + "=" * 72)
    print(f"  PROJECT AUDIT — Hebrew Recipe NLP (NLP 2025a)  •  {now}")
    print("=" * 72)

    # ── Summary counts ────────────────────────────────────────────────────
    p0 = sum(1 for a in actions if a["priority"] == "P0")
    p1 = sum(1 for a in actions if a["priority"] == "P1")
    p2 = sum(1 for a in actions if a["priority"] == "P2")
    total_size = human_size(scan["all_files_size"])

    print(f"\n  Files on disk : {scan['all_files_count']:,}  ({total_size})")
    print(f"  Action items  : 🔴 {p0} blocker(s)  🟡 {p1} important  🟢 {p2} cleanup")

    # ── HF summary ────────────────────────────────────────────────────────
    print(f"\n  HuggingFace (@{HF_USERNAME})")
    print(f"    Datasets : {len(hf.get('datasets', []))}")
    print(f"    Models   : {len(hf.get('models', []))}")
    if hf.get("errors"):
        for e in hf["errors"]:
            print(f"    ⚠  {e}")
    for ds in hf.get("datasets", []):
        vis = "🔒 PRIVATE" if ds["private"] else "🌐 public"
        print(f"      📦 {ds['id']}  [{vis}]  last: {ds['last_modified'][:10]}")
    for m in hf.get("models", []):
        vis = "🔒 PRIVATE" if m["private"] else "🌐 public"
        print(f"      🤖 {m['id']}  [{vis}]  last: {m['last_modified'][:10]}")

    # ── Git summary ───────────────────────────────────────────────────────
    if git.get("available"):
        print(f"\n  Git")
        print(f"    Branch       : {git.get('current_branch', '?')}")
        print(f"    Last commit  : {git.get('last_commit', '?')}")
        print(f"    Remote       : {git.get('remote_url', '?')}")
        ab = git.get("ahead_behind")
        if ab:
            print(f"    Status       : {ab}")
        dirty = git.get("modified", [])
        untracked = git.get("untracked", [])
        if dirty:
            print(f"    Modified     : {len(dirty)} file(s) with uncommitted changes")
        if untracked:
            print(f"    Untracked    : {len(untracked)} untracked file(s)")
    else:
        print("\n  Git: not available / no .git found")

    # ── SLURM completeness table ──────────────────────────────────────────
    print("\n  SLURM Sbatch Coverage")
    print(f"  {'Model':<20} {'Dir':^5} {'Found':^6} {'Missing':^8} {'Complete':^10}")
    print("  " + "-" * 55)
    for slug, sm in scan["slurm_map"].items():
        total = len(EXPECTED_SBATCH_SUFFIXES)
        found = len(sm["found"])
        miss = len(sm["missing"])
        ok = "✅" if sm["dir_exists"] else "❌"
        bar = "█" * int(sm["completeness_pct"] / 10) + "░" * (10 - int(sm["completeness_pct"] / 10))
        pct = f"{sm['completeness_pct']:5.1f}%"
        print(f"  {slug:<20} {ok:^5} {found:^6} {miss:^8} {pct} {bar}")

    # ── Canonical files status ────────────────────────────────────────────
    missing_src = [r for r, i in scan["canonical_src"].items() if not i["exists"]]
    missing_scripts = [r for r, i in scan["canonical_scripts"].items() if not i["exists"]]
    print(f"\n  Canonical src files  : {len(scan['canonical_src']) - len(missing_src)}/{len(scan['canonical_src'])} present")
    for m in missing_src:
        print(f"    ❌ MISSING: {m}")
    print(f"  Canonical scripts    : {len(scan['canonical_scripts']) - len(missing_scripts)}/{len(scan['canonical_scripts'])} present")
    for m in missing_scripts:
        print(f"    ❌ MISSING: {m}")

    # ── Data files ────────────────────────────────────────────────────────
    print(f"\n  Required data files")
    for rel, info in scan["data_required"].items():
        icon = "✅" if info["exists"] else "❌"
        size = human_size(info["size"]) if info["exists"] else "MISSING"
        print(f"    {icon} {rel:<60} {size:>10}")

    print(f"\n  v2 data files (Track A output)")
    for rel, info in scan["data_v2"].items():
        icon = "✅" if info["exists"] else "⬜"
        size = human_size(info["size"]) if info["exists"] else "not yet"
        print(f"    {icon} {rel:<60} {size:>10}")

    # ── Results files ─────────────────────────────────────────────────────
    if scan["results_files"]:
        print(f"\n  Results files ({len(scan['results_files'])})")
        for r in sorted(scan["results_files"], key=lambda x: x["age_days"]):
            print(f"    📊 {r['path']}  ({human_size(r['size'])}, {r['age_days']:.0f}d ago)")

    # ── Version conflicts ─────────────────────────────────────────────────
    if scan["version_groups"]:
        print(f"\n  Version conflicts on disk ({len(scan['version_groups'])} group(s))")
        for base, versions in scan["version_groups"].items():
            newest = max(versions, key=lambda x: x["version"])
            print(f"    Base: {base}")
            for v in versions:
                tag = " ← KEEP (newest)" if v["version"] == newest["version"] else " ← review/delete"
                print(f"      v{v['version']} : {v['path']}{tag}")

    # ── Debris ────────────────────────────────────────────────────────────
    if scan["debris"]:
        print(f"\n  Debris / backup files ({len(scan['debris'])})")
        for d in scan["debris"]:
            print(f"    🗑  {d['path']}  ({human_size(d['size'])}, {d['age_days']:.0f}d old)")

    # ── Action plan ───────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  ACTION PLAN  ({len(actions)} items)")
    print(f"{'=' * 72}")

    last_prio = None
    for i, a in enumerate(actions, 1):
        if a["priority"] != last_prio:
            prio_label = {
                "P0": "BLOCKERS — fix before anything else",
                "P1": "IMPORTANT — fix before submission",
                "P2": "CLEANUP — nice-to-have",
            }.get(a["priority"], a["priority"])
            print(f"\n  {PRIORITY_EMOJI.get(a['priority'], '')} {prio_label}")
            last_prio = a["priority"]
        cat = CATEGORY_LABELS.get(a["category"], a["category"])
        print(f"\n  [{i:02d}] {cat}")
        print(f"       Item    : {a['item']}")
        print(f"       Detail  : {a['detail']}")
        print(f"       Action  : {a['suggestion']}")

    print(f"\n{'=' * 72}\n")


def write_markdown_report(
    scan: Dict, git: Dict, hf: Dict, actions: List[Dict], out_path: Path
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    p0 = sum(1 for a in actions if a["priority"] == "P0")
    p1 = sum(1 for a in actions if a["priority"] == "P1")
    p2 = sum(1 for a in actions if a["priority"] == "P2")

    lines = [
        f"# Project Audit — Hebrew Recipe NLP (NLP 2025a)",
        f"*Generated: {now}*",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Files on disk | {scan['all_files_count']:,} ({human_size(scan['all_files_size'])}) |",
        f"| 🔴 Blockers | {p0} |",
        f"| 🟡 Important | {p1} |",
        f"| 🟢 Cleanup | {p2} |",
        f"| HF Datasets | {len(hf.get('datasets', []))} |",
        f"| HF Models | {len(hf.get('models', []))} |",
        f"| Git branch | {git.get('current_branch', 'n/a')} |",
        f"| Last commit | {git.get('last_commit', 'n/a')} |",
        "",
    ]

    # HF section
    lines += ["## HuggingFace Assets", ""]
    if hf.get("errors"):
        for e in hf["errors"]:
            lines.append(f"> ⚠️ {e}")
        lines.append("")
    if hf.get("datasets"):
        lines += ["### Datasets", ""]
        lines += ["| Repo | Visibility | Last Modified | Downloads |", "|------|-----------|--------------|-----------|"]
        for ds in hf["datasets"]:
            vis = "🔒 Private" if ds["private"] else "🌐 Public"
            lines.append(f"| [{ds['id']}](https://huggingface.co/datasets/{ds['id']}) | {vis} | {ds['last_modified'][:10]} | {ds['downloads']} |")
        lines.append("")
    if hf.get("models"):
        lines += ["### Models", ""]
        lines += ["| Repo | Visibility | Pipeline | Last Modified |", "|------|-----------|----------|--------------|"]
        for m in hf["models"]:
            vis = "🔒 Private" if m["private"] else "🌐 Public"
            lines.append(f"| [{m['id']}](https://huggingface.co/{m['id']}) | {vis} | {m['pipeline_tag'] or '—'} | {m['last_modified'][:10]} |")
        lines.append("")
    if hf.get("version_pairs"):
        lines += ["### ⚠️ Version Conflicts on HuggingFace", ""]
        for vp in hf["version_pairs"]:
            lines.append(f"- **{vp['base']}**: {' vs '.join(vp['versions'])}")
        lines.append("")

    # SLURM section
    lines += ["## SLURM Sbatch Coverage", ""]
    lines += ["| Model | Dir | Found | Missing | Complete |", "|-------|-----|-------|---------|----------|"]
    for slug, sm in scan["slurm_map"].items():
        ok = "✅" if sm["dir_exists"] else "❌"
        miss_str = ", ".join(sm["missing"][:3]) + ("…" if len(sm["missing"]) > 3 else "")
        lines.append(f"| {slug} | {ok} | {len(sm['found'])} | {len(sm['missing'])} ({miss_str}) | {sm['completeness_pct']}% |")
    lines.append("")

    # Canonical files section
    lines += ["## Canonical File Status (§10.1)", ""]
    lines += ["| File | Status | Size | Last Modified |", "|------|--------|------|--------------|"]
    for rel, info in {**scan["canonical_src"], **scan["canonical_scripts"]}.items():
        status = "✅" if info["exists"] else "❌ MISSING"
        size = human_size(info["size"]) if info["exists"] else "—"
        age = f"{info['age_days']:.0f}d ago" if info["age_days"] > 0 else "—"
        lines.append(f"| `{rel}` | {status} | {size} | {age} |")
    lines.append("")

    # Data files section
    lines += ["## Data File Status (§10.2)", ""]
    lines += ["### Required (v1)", ""]
    lines += ["| File | Status | Size |", "|------|--------|------|"]
    for rel, info in scan["data_required"].items():
        status = "✅" if info["exists"] else "❌ MISSING"
        size = human_size(info["size"]) if info["exists"] else "—"
        lines.append(f"| `{rel}` | {status} | {size} |")
    lines.append("")
    lines += ["### Track A Output (v2)", ""]
    lines += ["| File | Status | Size |", "|------|--------|------|"]
    for rel, info in scan["data_v2"].items():
        status = "✅" if info["exists"] else "⬜ not yet"
        size = human_size(info["size"]) if info["exists"] else "—"
        lines.append(f"| `{rel}` | {status} | {size} |")
    lines.append("")

    # Version conflicts on disk
    if scan["version_groups"]:
        lines += ["## Version Conflicts on Disk", ""]
        for base, versions in scan["version_groups"].items():
            newest = max(versions, key=lambda x: x["version"])
            lines.append(f"**`{base}`**")
            for v in versions:
                tag = " ← **KEEP**" if v["version"] == newest["version"] else " ← review/delete"
                lines.append(f"- `{v['path']}` (v{v['version']}, {human_size(v['size'])}, {v['age_days']:.0f}d ago){tag}")
        lines.append("")

    # Debris
    if scan["debris"]:
        lines += ["## Debris / Backup Files", ""]
        lines += ["| File | Size | Age | Action |", "|------|------|-----|--------|"]
        for d in scan["debris"]:
            lines.append(f"| `{d['path']}` | {human_size(d['size'])} | {d['age_days']:.0f}d | Delete or archive |")
        lines.append("")

    # Git status
    lines += ["## Git Status", ""]
    if git.get("available"):
        lines += [
            f"- **Remote**: {git.get('remote_url', '?')}",
            f"- **Branch**: {git.get('current_branch', '?')}",
            f"- **Last commit**: {git.get('last_commit', '?')} (`{git.get('last_commit_date', '?')[:10]}`)",
        ]
        ab = git.get("ahead_behind")
        if ab:
            lines.append(f"- **Sync**: {ab}")
        lines.append("")
        lines += ["### Files that MUST be committed", ""]
        lines += ["| File | Status |", "|------|--------|"]
        for rel, status in git.get("must_be_committed_status", {}).items():
            icon = "✅" if status == "tracked" else "❌"
            lines.append(f"| `{rel}` | {icon} {status} |")
        lines.append("")
    else:
        lines.append("*Git not available or no `.git` directory found.*")
        lines.append("")

    # Action plan
    lines += ["---", "## Action Plan", ""]
    last_prio = None
    for i, a in enumerate(actions, 1):
        if a["priority"] != last_prio:
            prio_label = {
                "P0": "🔴 Blockers — fix before anything else",
                "P1": "🟡 Important — fix before submission",
                "P2": "🟢 Cleanup — nice-to-have",
            }.get(a["priority"], a["priority"])
            lines += [f"### {prio_label}", ""]
            last_prio = a["priority"]
        cat = CATEGORY_LABELS.get(a["category"], a["category"])
        lines += [
            f"#### [{i:02d}] {cat}: `{a['item']}`",
            f"- **Issue**: {a['detail']}",
            f"- **Fix**: `{a['suggestion']}`",
            "",
        ]

    lines += ["---", f"*Audit script: `project_audit.py` — update `MASTER_PLAN_v8_1.txt` with findings*", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  ✍  Markdown report saved to: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Audit the Hebrew Recipe NLP project: SLURM, HuggingFace, and Git."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", None),
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Skip HuggingFace checks",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Skip git checks",
    )
    parser.add_argument(
        "--report-md",
        default="audit_report.md",
        help="Path for the Markdown report output (default: audit_report.md)",
    )
    parser.add_argument(
        "--json",
        default="audit_snapshot.json",
        help="Path for the JSON snapshot (default: audit_snapshot.json)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"\n  Scanning project root: {root}")

    if not root.exists():
        print(f"  ERROR: Root directory does not exist: {root}")
        sys.exit(1)

    # ── 1. Scan local files ───────────────────────────────────────────────
    print("  [1/4] Scanning local file system...")
    scan = scan_project(root)

    # ── 2. Git audit ──────────────────────────────────────────────────────
    if args.no_git:
        git = {"available": False, "note": "Skipped via --no-git"}
    else:
        print("  [2/4] Auditing git status...")
        git = audit_git(root)

    # ── 3. HuggingFace audit ──────────────────────────────────────────────
    if args.no_hf:
        hf: Dict[str, Any] = {"available": False, "datasets": [], "models": [],
                               "version_pairs": [], "errors": ["Skipped via --no-hf"]}
    else:
        print(f"  [3/4] Querying HuggingFace (@{HF_USERNAME})...")
        hf = audit_huggingface(args.hf_token)

    # ── 4. Cross-reference ────────────────────────────────────────────────
    print("  [4/4] Cross-referencing and generating action plan...")
    actions = cross_reference(scan, git, hf)

    # ── Terminal report ───────────────────────────────────────────────────
    print_terminal_report(scan, git, hf, actions)

    # ── Markdown report ───────────────────────────────────────────────────
    write_markdown_report(scan, git, hf, actions, Path(args.report_md))

    # ── JSON snapshot ─────────────────────────────────────────────────────
    snapshot = {
        "generated": datetime.now().isoformat(),
        "root": str(root),
        "scan": scan,
        "git": git,
        "hf": hf,
        "actions": actions,
    }
    Path(args.json).write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    print(f"  📦 JSON snapshot saved to: {args.json}")
    print()


if __name__ == "__main__":
    main()
