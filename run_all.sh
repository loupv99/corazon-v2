#!/usr/bin/env bash
# Corazón v2: todos los comandos en orden. Ajustar las rutas de entrada (en Kaggle las arma el notebook 01).
set -euo pipefail

NIH_CSV=${NIH_CSV:-/kaggle/input/data/Data_Entry_2017.csv}
NIH_ROOT=${NIH_ROOT:-/kaggle/input/data}
TEST_LIST=${TEST_LIST:-/kaggle/input/data/test_list.txt}
JSRT_DIR=${JSRT_DIR:?"definir JSRT_DIR con las radiografías JSRT limpias"}
RAW=${RAW:-/tmp/raw}
DATA=${DATA:-data/processed}
mkdir -p "$RAW"

# 0. Descargas
[ -f "$RAW/ChestX-Ray8.csv" ] || wget -q -O "$RAW/ChestX-Ray8.csv" \
  https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/OriginalResolution/ChestX-Ray8.csv
[ -f "$RAW/points.zip" ] || wget -q -O "$RAW/points.zip" "https://zenodo.org/records/7056076/files/points.zip?download=1"

# 1. Preparación (una vez)
python -m src.prepare_data --nih-csv "$NIH_CSV" --nih-root "$NIH_ROOT" --test-list "$TEST_LIST" \
  --chexmask "$RAW/ChestX-Ray8.csv" --scr-points "$RAW/points.zip" --scr-images "$JSRT_DIR" --out "$DATA"

# 2. E1: etiquetas débiles vs. doradas
python -m src.train_seg --data "$DATA" --run E1b_debil_rca07 --masks rca07 --epochs 30
python -m src.train_seg --data "$DATA" --run E1a_debil_todas --masks all --epochs 30
python -m src.train_seg --data "$DATA" --run E1c_rca07_ft_scr --train-source scr \
  --init runs/E1b_debil_rca07/best.pt --epochs 30 --lr 2e-4 --bs 8

# 3. E2–E6: pendientes (clasificador ResNet-50, Grad-CAM, fotos con celular)
