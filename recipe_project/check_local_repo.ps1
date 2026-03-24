# =============================================================================
# LOCAL REPO AUDIT (PowerShell) - ASCII Safe Version
# =============================================================================

$ErrorActionPreference = "Stop"

$RED = "`e[0;31m"
$GREEN = "`e[0;32m"
$YELLOW = "`e[1;33m"
$CYAN = "`e[0;36m"
$NC = "`e[0m"
$BOLD = "`e[1m"

$script:pass = 0
$script:fail = 0
$script:warn = 0

function Check-File {
    param([string]$path, [string]$desc)
    if (Test-Path -Path $path -PathType Leaf) {
        Write-Host "  ${GREEN}[OK]${NC} $path"
        $script:pass++
    } else {
        Write-Host "  ${RED}[MISSING]${NC} $path  - $desc"
        $script:fail++
    }
}

function Check-Dir {
    param([string]$path, [string]$desc)
    if (Test-Path -Path $path -PathType Container) {
        Write-Host "  ${GREEN}[OK]${NC} $path/"
        $script:pass++
    } else {
        Write-Host "  ${RED}[MISSING DIR]${NC} $path/  - $desc"
        $script:fail++
    }
}

function Check-File-Warn {
    param([string]$path, [string]$desc)
    if (Test-Path -Path $path -PathType Leaf) {
        Write-Host "  ${GREEN}[OK]${NC} $path"
        $script:pass++
    } else {
        Write-Host "  ${YELLOW}[OPTIONAL]${NC} $path  - $desc"
        $script:warn++
    }
}

Write-Host ""
Write-Host "${BOLD}=========================================="
Write-Host "LOCAL REPO AUDIT (V7 Master Plan)"
Write-Host "==========================================${NC}"
Write-Host "Directory: $((Get-Location).Path)"
Write-Host "Date: $(Get-Date)"
Write-Host ""

if (!(Test-Path "run_pipeline.py") -and (!(Test-Path "src\__init__.py"))) {
    Write-Host "${RED}ERROR: Not in project root! cd to your recipe_project directory first.${NC}"
    exit 1
}

Write-Host "${CYAN}1. Top-level files${NC}"
Check-File "run_pipeline.py" "Pipeline runner"
Check-File ".gitignore" "Git ignore rules"
Check-File "README.md" "Project README"
Check-File "Pipfile" "Python dependencies"
Write-Host ""

Write-Host "${CYAN}2. Config${NC}"
Check-Dir "config" "Configuration directory"
Check-File "config/config.yaml" "Main project config"
Write-Host ""

Write-Host "${CYAN}3. Source: Core modules${NC}"
Check-File "src/__init__.py" "Main package init"
Write-Host ""

Write-Host "${CYAN}4. Source: Teacher labeling${NC}"
Check-Dir "src/teacher_labeling" "Teacher labeling module"
Check-File "src/teacher_labeling/__init__.py" "Module init"
Check-File "src/teacher_labeling/generate_labels.py" "3-pass silver label generation"
Write-Host ""

Write-Host "${CYAN}5. Source: Preprocessing${NC}"
Check-Dir "src/preprocessing" "Preprocessing module"
Check-File "src/preprocessing/__init__.py" "Module init"
Check-File "src/preprocessing/prepare_data.py" "Original preprocessing"
Check-File "src/preprocessing/prepare_data_merged.py" "V7 merged preprocessing"
Write-Host ""

Write-Host "${CYAN}6. Source: Models${NC}"
Check-Dir "src/models" "Models module"
Check-File "src/models/__init__.py" "Module init"
Check-File "src/models/train_student.py" "Softmax model training"

if (Test-Path "src/models/train_student.py") {
    if (Select-String -Path "src/models/train_student.py" -Pattern "class\.weights|focal\.loss|FocalLoss|WeightedTrainer" -Quiet) {
        Write-Host "  ${GREEN}  -> has class weights / focal loss support${NC}"
    } else {
        Write-Host "  ${YELLOW}  -> WARNING: may need --class-weights and --focal-loss flags added${NC}"
        $script:warn++
    }
}

Check-File "src/models/joint_model.py" "BERT + CRF model"
Check-File "src/models/train_joint.py" "CRF training loop"

if (Test-Path "src/models/joint_model.py") {
    if (Select-String -Path "src/models/joint_model.py" -Pattern "intent" -Quiet) {
        Write-Host "  ${YELLOW}  -> WARNING: joint_model.py still contains intent references${NC}"
        $script:warn++
    } else {
        Write-Host "  ${GREEN}  -> clean - no intent conditioning${NC}"
    }
}

if (Test-Path "src/models/train_joint.py") {
    if (Select-String -Path "src/models/train_joint.py" -Pattern "intent_weight|LabelSmoothCE|labels_intent" -Quiet) {
        Write-Host "  ${YELLOW}  -> WARNING: train_joint.py still has intent conditioning${NC}"
        $script:warn++
    } else {
        Write-Host "  ${GREEN}  -> clean - no intent conditioning${NC}"
    }
}
Write-Host ""

Write-Host "${CYAN}7. Source: Utils${NC}"
Check-Dir "src/utils" "Utils module"
Check-File "src/utils/__init__.py" "Module init"
Check-File "src/utils/class_weights.py" "FocalLoss weights"
Write-Host ""

Write-Host "${CYAN}8. Source: Baselines${NC}"
Check-Dir "src/baselines" "Baselines module"
Check-File "src/baselines/__init__.py" "Module init"
Check-File "src/baselines/run_baselines.py" "Baselines"
Write-Host ""

Write-Host "${CYAN}9. Source: Evaluation${NC}"
Check-Dir "src/evaluation" "Evaluation module"
Check-File "src/evaluation/__init__.py" "Module init"
Check-File "src/evaluation/evaluate.py" "Evaluation script"
Write-Host ""

Write-Host "${CYAN}10. Scripts${NC}"
Check-File "scripts/verify_data_integrity.py" "Leakage check"
Check-File "scripts/evaluate_teacher.py" "Teacher upper bound"
Check-File-Warn "scripts/run_all_ablations.sh" "Ablation submitter"
Write-Host ""

Write-Host "${CYAN}11. SLURM scripts${NC}"
Check-Dir "scripts/slurm" "SLURM scripts directory"
Check-File "scripts/slurm/setup_cluster.sh" "Cluster setup"
Check-File "scripts/slurm/run_experiments.sh" "Experiment runner"
Check-File "scripts/slurm/test_gpu.sbatch" "GPU test"
Check-File "scripts/slurm/preprocess_data.sbatch" "Preprocessing"

Write-Host ""
Write-Host "${CYAN}  Model training sbatch files:${NC}"
Check-File "scripts/slurm/train_alephbert.sbatch" "AlephBERT"
Check-File "scripts/slurm/train_mbert.sbatch" "mBERT"
Check-File "scripts/slurm/train_hebert.sbatch" "HeBERT"
Check-File "scripts/slurm/train_xlmr.sbatch" "XLM-RoBERTa"

if (!(Test-Path "scripts/slurm/train_xlmr.sbatch") -and (Test-Path "scripts/slurm/train_xlm_roberta.sbatch")) {
    Write-Host "  ${GREEN}  -> found as train_xlm_roberta.sbatch${NC}"
    $script:pass++
    $script:fail--
}

Check-File "scripts/slurm/train_dictabert.sbatch" "DictaBERT"
Check-File-Warn "scripts/slurm/train_dicta_best.sbatch" "DictaBERT-Large"
if (!(Test-Path "scripts/slurm/train_dicta_best.sbatch")) {
    $altNames = @("train_dictabert_large.sbatch", "train_dicta_large.sbatch", "train_twostep_large.sbatch")
    foreach ($alt in $altNames) {
        if (Test-Path "scripts/slurm/$alt") {
            Write-Host "  ${GREEN}  -> found as $alt${NC}"
            break
        }
    }
}
Check-File "scripts/slurm/train_joint.sbatch" "DictaBERT + CRF"

Write-Host ""
Write-Host "${CYAN}  Evaluation sbatch files:${NC}"
Check-File "scripts/slurm/evaluate_model.sbatch" "Evaluation"
Check-File-Warn "scripts/slurm/eval_pipeline.sbatch" "Eval pipeline"

Write-Host ""
Write-Host "${CYAN}  Ablation sbatch files:${NC}"
Check-File "scripts/slurm/ablation_data_size.sbatch" "Ablation: data size"
$ablations = @("ablation_downsample", "ablation_bio_io", "ablation_enriched", "ablation_confidence")
foreach ($abl in $ablations) {
    Check-File-Warn "scripts/slurm/${abl}.sbatch" "Ablation: ${abl}"
}
Write-Host ""

Write-Host "${CYAN}12. YouTube collector${NC}"
Check-Dir "youtube_collector" "YouTube collector module"
Check-File "youtube_collector/collect.py" "Main collector script"
Check-File "youtube_collector/config.yaml" "Collection config"
Check-File "youtube_collector/channels.yaml" "Channel list"
Write-Host ""

Write-Host "${CYAN}13. Data directory (.gitignored)${NC}"
if (Test-Path "data" -PathType Container) {
    Write-Host "  ${GREEN}[OK]${NC} data/ exists locally"
    $subDirs = @("raw_youtube", "silver_labels", "processed", "gold_validation")
    foreach ($sub in $subDirs) {
        if (Test-Path "data/$sub" -PathType Container) {
            $count = @(Get-ChildItem -Path "data/$sub" -File -ErrorAction SilentlyContinue).Count
            Write-Host "  ${GREEN}[OK]${NC} data/$sub/ ($count files)"
        } else {
            Write-Host "  ${YELLOW}[?]${NC} data/$sub/ (not present locally - OK if on SLURM)"
        }
    }
} else {
    Write-Host "  ${YELLOW}[?]${NC} data/ not present locally - expected if .gitignored"
}
Write-Host ""

Write-Host "${BOLD}=========================================="
Write-Host "SUMMARY"
Write-Host "==========================================${NC}"
Write-Host "  ${GREEN}Pass:${NC}    $script:pass"
Write-Host "  ${RED}Missing:${NC} $script:fail"
Write-Host "  ${YELLOW}Warning:${NC} $script:warn"
Write-Host ""

if ($script:fail -eq 0) {
    Write-Host "${GREEN}${BOLD}ALL REQUIRED FILES PRESENT${NC}"
} else {
    Write-Host "${RED}${BOLD}$script:fail REQUIRED FILES MISSING - fix before proceeding${NC}"
}
Write-Host ""