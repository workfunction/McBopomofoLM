#!/bin/bash
# One-time setup of the offline evaluation environment under $EVAL_WORK (default ~/.cache/mcbpmflm-eval).
set -euo pipefail
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$TOOLS/../../.." && pwd)
WORK=${EVAL_WORK:-$HOME/.cache/mcbpmflm-eval}; mkdir -p "$WORK/models/sloth" "$WORK/models/q35" "$WORK/g2pw-model"
python3 -m venv "$WORK/venv"; "$WORK/venv/bin/pip" install -q -r "$TOOLS/requirements.txt"
HF=https://huggingface.co
SLOTH_REV=e7d13c9c451dc01ab9546be53c574f4dbe0fdd54   # Luigi/sloth-ime-models (Apache-2.0)
for f in train_slothe_ternary.py gate_slothe_ternary.py slothe_config.json syl_vocab.json syl2legal.npz slothe.pt; do
  curl -fsSL -o "$WORK/models/sloth/$f" "$HF/Luigi/sloth-ime-models/resolve/$SLOTH_REV/$f"; done
for f in config.json generation_config.json tokenizer.json model.safetensors; do
  curl -fsSL -o "$WORK/models/q35/$f" "$HF/Luigi/sloth-ime-models/resolve/$SLOTH_REV/pred_q35_60m/$f"; done
"$WORK/venv/bin/hf" download Funmula/0c-sequence-d512 --revision d512-pf512-20260926 --local-dir "$WORK/models/d512"   # MIT
curl -fsSL -o "$WORK/g2pw-model/m.zip" https://storage.googleapis.com/esun-ai/g2pW/G2PWModel-v2-onnx.zip
(cd "$WORK/g2pw-model" && unzip -q -o m.zip && rm m.zip)
(cd "$REPO/Source/Data" && make data.txt)
"$TOOLS/build_walk_cli.sh"
echo "ready: cd $WORK && $WORK/venv/bin/python -I $TOOLS/val_walk.py $WORK"
