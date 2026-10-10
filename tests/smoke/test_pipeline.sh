#!/usr/bin/env bash
# End-to-end smoketest: synthetic data -> train cls+seg for each backbone -> eval -> infer/export -> compare.
#   bash tests/smoke/test_pipeline.sh [backbone ...]     (default: dinov2_s dinov2_b)
set -euo pipefail
cd "$(dirname "$0")/../.."
BACKBONES=${*:-dinov2_s dinov2_b}

python -m tests.smoke.test_env
[ -f "$DINO_DATA/synthetic/classes.txt" ] || python -m dino.data.synthetic --out "$DINO_DATA/synthetic"

for bb in $BACKBONES; do
  for task in cls seg; do
    name=smoke_${task}_${bb}
    python -m dino.train --config configs/smoke_${task}.yaml --set backbone=$bb name=$name
    python -m dino.evaluate --run "$DINO_RUNS/$name"
  done
done

first=${BACKBONES%% *}
python -m dino.infer --run "$DINO_RUNS/smoke_cls_$first" --input "$DINO_DATA/synthetic/val" --out "$DINO_RUNS/smoke_cls_$first/preds"
python -m dino.infer --run "$DINO_RUNS/smoke_seg_$first" --input "$DINO_DATA/synthetic/val" --out "$DINO_RUNS/smoke_seg_$first/fp_scene"
python -m tests.smoke.check_fp_export "$DINO_RUNS/smoke_seg_$first/fp_scene"
python -m dino.compare --glob 'smoke_*'
echo "SMOKE PIPELINE OK"
