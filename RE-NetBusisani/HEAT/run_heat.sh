#!/usr/bin/env bash
# HEAT Run Script — runs training then evaluation then visualization
# Usage: bash run_heat.sh ICEWS18

DATASET=${1:-ICEWS18}

echo "========================================"
echo " HEAT Pipeline — Dataset: $DATASET"
echo "========================================"

cd "$(dirname "$0")"

echo ""
echo "[1/3] Training HEAT..."
conda run -n renet python3 train.py \
    -d "$DATASET" \
    --n-hidden 200 \
    --time-dim 32 \
    --n-local-heads 4 \
    --n-global-heads 4 \
    --seq-len 10 \
    --dropout 0.3 \
    --lr 1e-3 \
    --batch-size 512 \
    --max-epochs 30 \
    --valid-every 2 \
    --val-samples 2000 \
    --alpha 0.1 \
    --beta 0.05 \
    2>&1 | tee "results/${DATASET}/train.log"

echo ""
echo "[2/3] Running evaluation..."
conda run -n renet python3 test.py \
    -d "$DATASET" \
    --uncertainty \
    2>&1 | tee "results/${DATASET}/test.log"

echo ""
echo "[3/3] Generating visualizations..."
conda run -n renet python3 visualize.py \
    -d "$DATASET" \
    2>&1 | tee "results/${DATASET}/viz.log"

echo ""
echo "✅ Done! Results in: results/$DATASET/"
echo "   Figures in: results/$DATASET/figures/"
