#!/bin/bash
LOG_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/logs/slurm_output/dictabert_crf"

printf "%-8s | %-12s | %-10s | %-8s | %-15s\n" "JOBID" "VARIANT" "STATUS" "BEST F1" "LATEST EPOCH"
echo "--------------------------------------------------------------------------------"

# Added 305036 to the list for your new run
IDS=(305035 305027 304948 304949 305028 305029 305030 305036)

for ID in "${IDS[@]}"; do
    STATUS=$(squeue -j $ID -h -o %T 2>/dev/null)
    [ -z "$STATUS" ] && STATUS="FINISHED"

    LOG=$(find $LOG_BASE -name "${ID}_*.out" | head -n 1)
    if [ -z "$LOG" ]; then
        printf "%-8s | %-12s | %-10s | %-8s | %-15s\n" "$ID" "Unknown" "$STATUS" "---" "No log found"
        continue
    fi

    NAME=$(basename "$LOG" | sed -E "s/${ID}_//" | sed 's/\.out//' | cut -c 1-12)
    
    # Robust numeric extraction: finds digits and dots, strips trailing punctuation
    BEST_F1=$(grep -oP "(New best F1=|\"best_f1\": )\K[0-9.]+" "$LOG" | tail -1 | sed 's/\.$//')
    [[ ! $BEST_F1 =~ ^[0-9.]+$ ]] && BEST_F1="0.0000"
    
    PROGRESS=$(grep -oP "Epoch \d+/\d+" "$LOG" | tail -1)
    [ -z "$PROGRESS" ] && PROGRESS="Starting..."

    printf "%-8s | %-12s | %-10s | %-8.4f | %-15s\n" "$ID" "$NAME" "$STATUS" "$BEST_F1" "$PROGRESS"
done
