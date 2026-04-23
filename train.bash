#!/bin/bash
set -e

wandb login

EXPERIMENTS=("baseline" "mobilenetv3_small" "efficientnet_b3" "canny" "no_vae")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  LeRobot ACT - Training all experiments"
echo "  $(date)"
echo "=========================================="

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "  Starting: $EXP"
    echo "  $(date)"
    echo "------------------------------------------"

    LOG_FILE="$LOG_DIR/${EXP}_$(date +%Y%m%d_%H%M%S).log"

    python train.py "$EXP" 2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: $EXP failed with exit code $EXIT_CODE"
        echo "Log: $LOG_FILE"
        echo ""
        read -p "Continue with next experiment? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting."
            exit 1
        fi
    else
        echo "DONE: $EXP"
    fi
done

echo ""
echo "=========================================="
echo "  All experiments finished"
echo "  $(date)"
echo "=========================================="