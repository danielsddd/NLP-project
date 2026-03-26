#!/bin/bash
# MASTER SUBMIT SCRIPT
# Usage: bash submit_all.sh <command>
DIR="$(cd "$(dirname "$0")" && pwd)"
BEST="dictabert"

case "$1" in
    preprocess)
        echo "=== Submitting ALL preprocessing ==="
        for s in "$DIR"/preprocessing/*.sbatch; do
            echo "  $(basename $s)"; sbatch "$s"
        done ;;

    main)
        echo "=== Submitting MAIN config (P2) for all models ==="
        for m in mbert hebert xlm_roberta alephbert dictabert dictabert_large; do
            echo "  $m/P2_add_focal"; sbatch "$DIR/$m/P2_add_focal.sbatch"
        done
        echo "  dictabert_crf/P2_crf_weighted"; sbatch "$DIR/dictabert_crf/P2_crf_weighted.sbatch"
        ;;

    progressive)
        echo "=== Submitting P0-P5 for $BEST + CRF ==="
        for s in "$DIR/$BEST"/P*.sbatch; do
            echo "  $(basename $s)"; sbatch "$s"
        done
        for s in "$DIR/dictabert_crf"/P*.sbatch; do
            echo "  CRF/$(basename $s)"; sbatch "$s"
        done ;;

    progressive_all)
        echo "=== Submitting P0-P5 for ALL models ==="
        for m in mbert hebert xlm_roberta alephbert dictabert dictabert_large dictabert_crf; do
            for s in "$DIR/$m"/P*.sbatch; do
                echo "  $m/$(basename $s)"; sbatch "$s"
            done
        done ;;

    ablations)
        echo "=== Submitting ablations for $BEST ==="
        for s in "$DIR/$BEST"/A*.sbatch; do
            echo "  $(basename $s)"; sbatch "$s"
        done
        for s in "$DIR/dictabert_crf"/A*.sbatch; do
            echo "  CRF/$(basename $s)"; sbatch "$s"
        done ;;

    ablations_all)
        echo "=== ALL ablations × ALL models ==="
        for m in mbert hebert xlm_roberta alephbert dictabert dictabert_large dictabert_crf; do
            for s in "$DIR/$m"/A*.sbatch; do
                echo "  $m/$(basename $s)"; sbatch "$s"
            done
        done ;;

    combos)
        echo "=== Submitting combo scripts for $BEST ==="
        for s in "$DIR/$BEST"/combo_*.sbatch; do
            echo "  $(basename $s)"; sbatch "$s"
        done
        for s in "$DIR/dictabert_crf"/combo_*.sbatch; do
            echo "  CRF/$(basename $s)"; sbatch "$s"
        done ;;

    evaluate)
        echo "=== Evaluating all models ==="
        for m in mbert hebert xlm_roberta alephbert dictabert dictabert_large dictabert_crf; do
            echo "  $m"; sbatch "$DIR/$m/evaluate_all.sbatch"
        done ;;

    everything)
        echo "⚠️  Submitting EVERYTHING in 5 seconds..."
        sleep 5

        # Step 1: Preprocessing
        echo ""
        bash "$0" preprocess
        echo ""
        echo "Waiting 30s for preprocessing to queue..."
        sleep 30

        # Step 2: Training (all variants)
        bash "$0" progressive_all
        bash "$0" ablations_all
        bash "$0" combos

        # Step 3: Evaluation (submitted now, will wait in queue)
        # These jobs will likely start after training finishes
        # because they need checkpoints to exist
        echo ""
        echo "Submitting evaluation jobs (will run after training)..."
        bash "$0" evaluate

        echo ""
        echo "=== ALL JOBS SUBMITTED ==="
        echo "Monitor with: bash submit_all.sh status"
        ;;

    status)
        squeue -u $USER -o "%.8i %.15j %.10P %.8T %.10M %.20R" | head -60
        echo ""; squeue -u $USER -h -o "%T" | sort | uniq -c | sort -rn ;;

    count)
        TOTAL=0
        for m in preprocessing mbert hebert xlm_roberta alephbert dictabert dictabert_large dictabert_crf; do
            C=$(find "$DIR/$m" -name "*.sbatch" 2>/dev/null | wc -l)
            echo "  $m: $C"; TOTAL=$((TOTAL + C))
        done
        echo "  TOTAL: $TOTAL scripts" ;;

    *)
        echo "Usage: bash submit_all.sh {preprocess|main|progressive|progressive_all|ablations|ablations_all|combos|evaluate|everything|status|count}" ;;
esac
