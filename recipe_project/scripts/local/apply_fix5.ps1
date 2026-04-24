<#
.SYNOPSIS
    Fix 5: Patches sbatch files to (1) complete the xlm_roberta -> xlmr rename
    and (2) point evaluate_all.sbatch files at per-model tokenized gold.

.DESCRIPTION
    Two stages, idempotent, dry-run by default:

    Stage 1: Replace every literal 'xlm_roberta' with 'xlmr' in
             scripts/slurm/xlmr/*.sbatch and README.md (~220 replacements).

    Stage 2: In each scripts/slurm/<model>/evaluate_all.sbatch, replace the
             hardcoded line:
                 GOLD="data/gold_validation/gold_final.jsonl"
             with a self-healing block that:
               - Derives the tokenized-gold filename from the model slug
               - Maps dictabert_crf -> dictabert (shares tokenizer)
               - Auto-runs prepare_gold_for_eval.py if missing
               - Fails loudly on missing/empty gold

.EXAMPLE
    # From recipe_project/ root:
    powershell -File scripts/local/apply_fix5.ps1                 # dry-run (default)
    powershell -File scripts/local/apply_fix5.ps1 -Apply          # actually write changes
#>

[CmdletBinding()]
param(
    [switch]$Apply   # default = dry-run; -Apply writes to disk
)

$ErrorActionPreference = 'Stop'
$mode = if ($Apply) { "APPLY" } else { "DRY-RUN" }

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Fix 5 Patcher  [$mode]" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# --- Sanity: must run from project root -----------------------------------
if (-not (Test-Path "src/preprocessing/prepare_data.py")) {
    Write-Host "ERROR: must run from recipe_project/ root" -ForegroundColor Red
    Write-Host "       (no src/preprocessing/prepare_data.py found)" -ForegroundColor Red
    exit 1
}

$xlmrDir = "scripts/slurm/xlmr"
if (-not (Test-Path $xlmrDir)) {
    Write-Host "ERROR: $xlmrDir does not exist. Did the rename commit land?" -ForegroundColor Red
    exit 1
}

# ==========================================================================
#  STAGE 1: xlm_roberta -> xlmr inside scripts/slurm/xlmr/
# ==========================================================================
Write-Host "`n--- STAGE 1: xlm_roberta -> xlmr in $xlmrDir ---" -ForegroundColor Yellow

$stage1Files = Get-ChildItem -Path "$xlmrDir\*" -File -Include "*.sbatch","*.md"
$stage1Total = 0
$stage1FileCount = 0

foreach ($f in $stage1Files) {
    $orig = [System.IO.File]::ReadAllText($f.FullName)
    # Count literal matches (SimpleMatch)
    $count = ([regex]::Matches($orig, [regex]::Escape("xlm_roberta"))).Count
    if ($count -eq 0) {
        continue
    }
    $stage1FileCount++
    $stage1Total += $count
    $rel = $f.FullName.Substring((Get-Location).Path.Length + 1)
    Write-Host ("  {0,-55}  {1,3} replacements" -f $rel, $count)

    if ($Apply) {
        $new = $orig.Replace("xlm_roberta", "xlmr")
        # Preserve UTF-8 no-BOM, preserve LF line endings
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($f.FullName, $new, $utf8NoBom)
    }
}

Write-Host ("Stage 1 summary: {0} replacements in {1} files" -f $stage1Total, $stage1FileCount) -ForegroundColor Green

# ==========================================================================
#  STAGE 2: evaluate_all.sbatch -> per-model tokenized gold
# ==========================================================================
Write-Host "`n--- STAGE 2: per-model tokenized gold in evaluate_all.sbatch ---" -ForegroundColor Yellow

# The new block that replaces the single GOLD= line.
# Uses a bash function to resolve slug (with dictabert_crf -> dictabert override).
# Uses HEREDOC-in-variable style compatible with all current sbatch structures.
$newGoldBlock = @'
# === Fix 5: per-model tokenized gold with auto-regeneration ===
# Derive slug from this script's parent folder name.
# dictabert_crf shares the dictabert tokenizer -> reuse its tokenized gold.
MODEL_SLUG="$(basename "$(dirname "$0")")"
if [ "$MODEL_SLUG" = "dictabert_crf" ]; then
    GOLD_SLUG="dictabert"
else
    GOLD_SLUG="$MODEL_SLUG"
fi
GOLD="data/gold_validation/gold_tokenized_${GOLD_SLUG}.jsonl"

# Auto-regenerate the tokenized gold if the file is missing or empty.
if [ ! -s "$GOLD" ]; then
    echo "Tokenized gold not found or empty: $GOLD"
    echo "Regenerating via scripts/local/prepare_gold_for_eval.py ..."
    python scripts/local/prepare_gold_for_eval.py --all \
        || { echo "FATAL: gold tokenization failed"; exit 1; }
fi

if [ ! -s "$GOLD" ]; then
    echo "FATAL: $GOLD still missing/empty after regeneration. Aborting." >&2
    exit 1
fi

echo "Gold file: $GOLD ($(wc -l < "$GOLD") records)"
# === end Fix 5 ===
'@

# Normalize line endings to LF for cluster
$newGoldBlock = $newGoldBlock -replace "`r`n", "`n"

$stage2Files = Get-ChildItem -Path "scripts/slurm" -Recurse -Filter "evaluate_all.sbatch" |
    Where-Object { $_.FullName -notmatch '[\\/]old[\\/]' }

$stage2PatchedCount = 0
$stage2SkippedCount = 0

foreach ($f in $stage2Files) {
    $rel = $f.FullName.Substring((Get-Location).Path.Length + 1)
    $orig = [System.IO.File]::ReadAllText($f.FullName)

    # Already patched? (idempotent check)
    if ($orig -match 'Fix 5: per-model tokenized gold') {
        Write-Host ("  {0,-60}  already patched (skip)" -f $rel) -ForegroundColor DarkGray
        $stage2SkippedCount++
        continue
    }

    # Look for the exact line (LF or CRLF tolerated)
    $pattern = '(?m)^GOLD="data/gold_validation/gold_final\.jsonl"\s*$'
    if ($orig -notmatch $pattern) {
        Write-Host ("  {0,-60}  GOLD= line not found! SKIP" -f $rel) -ForegroundColor Red
        $stage2SkippedCount++
        continue
    }

    # Use MatchEvaluator to avoid $1/$2 backref expansion in replacement
    $new = [regex]::Replace($orig, $pattern, { param($m) $newGoldBlock })

    Write-Host ("  {0,-60}  patched" -f $rel) -ForegroundColor Green
    $stage2PatchedCount++

    if ($Apply) {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        # Ensure LF line endings in the whole file
        $new = $new -replace "`r`n", "`n"
        [System.IO.File]::WriteAllText($f.FullName, $new, $utf8NoBom)
    }
}

# Special case: xlmr's evaluate_all.sbatch also needs CKPT_BASE / RESULT_BASE
# fixed (they still reference xlm_roberta). Stage 1 already did the substring
# replacement if we ran it, so this is implicit — but verify.
$xlmrEval = "scripts/slurm/xlmr/evaluate_all.sbatch"
if (Test-Path $xlmrEval) {
    $content = [System.IO.File]::ReadAllText($xlmrEval)
    if ($content -match 'xlm_roberta') {
        Write-Host "`n  WARNING: $xlmrEval still contains xlm_roberta after Stage 1" -ForegroundColor Red
        Write-Host "           (check encoding issues)" -ForegroundColor Red
    } else {
        Write-Host "`n  ${xlmrEval}: no residual xlm_roberta refs" -ForegroundColor ... Green
    }
}

Write-Host ("`nStage 2 summary: {0} files patched, {1} skipped" -f $stage2PatchedCount, $stage2SkippedCount) -ForegroundColor Green

# ==========================================================================
#  DONE
# ==========================================================================
# ==========================================================================
#  DONE
# ==========================================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
if ($Apply) {
    Write-Host "  APPLIED. Review with: git diff --stat" -ForegroundColor Green
    Write-Host "  Spot-check: git diff scripts/slurm/xlmr/P0_baseline.sbatch"
    Write-Host "  Then commit when satisfied."
} else {
    Write-Host "  DRY-RUN complete. Nothing written." -ForegroundColor Yellow
    Write-Host "  Re-run with -Apply to write changes:"
    Write-Host "    powershell -File scripts/local/apply_fix5.ps1 -Apply"
}
Write-Host "================================================================" -ForegroundColor Cyan