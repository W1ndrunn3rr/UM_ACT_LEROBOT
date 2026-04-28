#!/bin/bash
set -uo pipefail

if ! command -v wandb >/dev/null 2>&1; then
    echo "ERROR: wandb is not installed or not in PATH"
    exit 1
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "wandb login skipped: using existing local login if available."
fi

EXPERIMENTS=("baseline" "mobilenetv3_small" "efficientnet_b0" "canny" "no_vae")
LOG_DIR="logs"
STATUS_FILE="$LOG_DIR/train_status_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  LeRobot ACT - Training all experiments"
echo "  $(date)"
echo "=========================================="
echo "Run status log: $STATUS_FILE"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "  Starting: $EXP"
    echo "  $(date)"
    echo "------------------------------------------"

    LOG_FILE="$LOG_DIR/${EXP}_$(date +%Y%m%d_%H%M%S).log"

    if python train.py "$EXP" 2>&1 | tee "$LOG_FILE"; then
        echo "DONE: $EXP"
        echo "$(date '+%Y-%m-%d %H:%M:%S') | SUCCESS | $EXP | $LOG_FILE" >> "$STATUS_FILE"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXIT_CODE=${PIPESTATUS[0]}

        echo ""
        echo "ERROR: $EXP failed with exit code $EXIT_CODE"
        echo "Log: $LOG_FILE"
        echo ""
        echo "$(date '+%Y-%m-%d %H:%M:%S') | FAIL($EXIT_CODE) | $EXP | $LOG_FILE" >> "$STATUS_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "Continuing with next experiment..."
    fi
done

echo ""
echo "=========================================="
echo "  All experiments finished"
echo "  $(date)"
echo "=========================================="
echo "Succeeded: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Status log: $STATUS_FILE"

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
