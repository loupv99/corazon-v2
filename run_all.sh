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

# 2. E1 (y E3): etiquetas débiles vs. doradas, 3 semillas
for s in 0 1 2; do
  python -m src.train_seg --data "$DATA" --run E1b_debil_rca07_s$s --masks rca07 --epochs 30 --seed $s
  python -m src.train_seg --data "$DATA" --run E1a_debil_todas_s$s --masks all --epochs 30 --seed $s
  python -m src.train_seg --data "$DATA" --run E1c_rca07_ft_scr_s$s --train-source scr \
    --init runs/E1b_debil_rca07_s$s/best.pt --epochs 30 --lr 2e-4 --bs 8 --seed $s
done

# 3. E2 y E4: ResNet-50 (PA + AP y solo PA) y Grad-CAM, 3 semillas
for s in 0 1 2; do
  python -m src.classifier --data "$DATA" --run E2_resnet_paap_s$s --seed $s
  python -m src.classifier --data "$DATA" --run E4_resnet_pa_s$s --views PA --seed $s
  for r in E2_resnet_paap_s$s E4_resnet_pa_s$s; do
    python -m src.gradcam --data "$DATA" --ckpt runs/$r/best.pt --out runs/$r
  done
done

# 4. Tablas: E1, E2, E3, E4
python -m src.analisis --data "$DATA" --runs runs --out results

# 5. E5: fotos con celular. En Kaggle (notebook 03) se eligen las 100 placas; el resto corre en la laptop:
python -m src.fotos seleccionar --subset "$DATA/subset.csv" --nih-root "$NIH_ROOT" --out e5_seleccion
#   python -m src.fotos mostrar --sel e5_seleccion          (pantalla completa; se fotografía cada lámina)
#   fotos en fotos/<celular>_<condicion>/, luego:
#   python -m src.e5 --sel e5_seleccion --fotos fotos --runs runs --out results/e5   (--solo-tablas: rehace tablas)

# 6. E6: el lado derecho del ICT (lee los ict_unet.csv de E1b; sin GPU)
python -m src.e6 --runs runs --out results/e6

# 7. App: U-Net -> ONNX (semilla elegida por Dice de validación). Probar con: cd app && python -m http.server 8000
python -m src.exportar_onnx --ckpt runs/E1b_debil_rca07_s1/best.pt --out app/modelo/unet.onnx \
  --png app/ejemplos/con_cardiomegalia.png
