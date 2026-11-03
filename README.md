# Corazón v2 — medir el índice cardiotorácico como el radiólogo

Proyecto final de MIA-203 Redes Neuronales y Aprendizaje Profundo (UNI FIIS, 2026-II).

Una U-Net segmenta pulmones y corazón en la radiografía de tórax y, con esos contornos, se calcula el índice
cardiotorácico (ICT). Se compara con una CNN de caja negra (ResNet-50) para detectar cardiomegalia, usando etiquetas
débiles (CheXmask) y doradas (SCR), y se mide la robustez a la proyección AP y a fotos tomadas con celular.

**Estado (2026-11-02):** día 1 corrido en Kaggle ([results/kaggle-v1/](results/kaggle-v1/LEEME.md)). Día 2 listo
para correr (notebook 02: semillas, ResNet-50, Grad-CAM, E1–E4). Fotos con celular (E5), cm (E6) y app: pendientes.

## Estructura

```
src/
  nih.py           subconjunto de ChestX-ray14 y partición por paciente
  chexmask.py      máscaras débiles (RLE de CheXmask)
  scr.py           contornos humanos de SCR (etiqueta dorada) sobre radiografías JSRT limpias
  prepare_data.py  arma todos los arrays de entrenamiento (una sola vez)
  ctr.py           índice cardiotorácico: ICT, MRD, MLD, ID (y en cm)
  unet.py          U-Net propia (reutilizada del examen parcial del grupo)
  dataset.py · losses.py · metrics.py · train_seg.py
  classifier.py    ResNet-50 preentrenada (caja negra), E2 y E4
  gradcam.py       Grad-CAM con hooks + energía del mapa dentro del corazón
  analisis.py      tablas E1 (semillas), E2, E3 (ICC, Bland-Altman), E4 (por vista, fuga de vista, Grad-CAM)
notebooks/
  01_kaggle_datos_y_unet.ipynb   día 1 en Kaggle ("Save & Run All", laptop apagada)
  02_kaggle_semillas_resnet_gradcam.ipynb   día 2: usa el Output del día 1, las dos GPU en paralelo
data/README.md     cómo obtener cada fuente de datos (no se suben datos)
results/           tablas y figuras del artículo
run_all.sh         todos los comandos en orden
```

## Datos

Ver [data/README.md](data/README.md). Resumen: NIH ChestX-ray14 (imágenes y etiquetas), CheXmask (máscaras
débiles, CC BY 4.0), SCR (contornos humanos, CC BY 4.0) y radiografías JSRT limpias.

## Reproducir

1. **Kaggle** (recomendado): abrir
   `notebooks/01_kaggle_datos_y_unet.ipynb`, seguir las instrucciones de la primera celda y "Save & Run All".
2. **Local / Colab:** `pip install -r requirements.txt` y seguir `run_all.sh`.

Pruebas sin GPU: `python -m src.ctr` (caso con respuesta conocida) y `python -m src.unet` (formas de salida).

## Semillas y hardware

Semilla 0 por defecto (`--seed`). Cada corrida guarda en `runs/<run>/results.json` la GPU usada y los minutos de
entrenamiento (requisito R6).

## Atribución de datos

- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC · Wang et al., CVPR 2017 · proveedor: NIH Clinical
  Center.
- CheXmask: Gaggion et al., Scientific Data 2024, doi 10.1038/s41597-024-03358-1 (CC BY 4.0).
- SCR: van Ginneken et al., Medical Image Analysis 2006 · Zenodo 7056076 (CC BY 4.0).
- JSRT: Shiraishi et al., AJR 2000.
