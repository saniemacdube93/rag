#!/usr/bin/env bash
# Run full HEAT pipeline (train → eval → visualize) for all datasets sequentially.
# Option A: Quick validation run — 5 epochs, batch 1024
set -e

export PYTHONUNBUFFERED=1

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate renet

DATASETS=("ICEWS18" "ICEWS14" "GDELT" "WIKI")

cd "$(dirname "$0")"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "  HEAT Pipeline — Dataset: $DS"
    echo "========================================"

    mkdir -p "results/$DS"

    echo "[1/3] Training $DS..."
    python3 -u train.py \
        -d "$DS" \
        --n-hidden 200 \
        --time-dim 32 \
        --n-local-heads 4 \
        --n-global-heads 4 \
        --seq-len 10 \
        --dropout 0.3 \
        --lr 1e-3 \
        --batch-size 1024 \
        --max-epochs 5 \
        --valid-every 1 \
        --val-samples 1000 \
        --alpha 0.1 \
        --beta 0.05 \
        2>&1 | tee "results/$DS/train.log"

    echo "[2/3] Evaluating $DS..."
    python3 -u test.py \
        -d "$DS" \
        --uncertainty \
        2>&1 | tee "results/$DS/test.log"

    echo "[3/3] Visualizing $DS..."
    python3 -u visualize.py \
        -d "$DS" \
        2>&1 | tee "results/$DS/viz.log"

    echo "✅ $DS done!"
done

echo ""
echo "🎉 All datasets complete! Results in: results/"
