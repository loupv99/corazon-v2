# Kaggle Version #1 — notebook 01 (2026-11-02)

GPU Tesla T4 (1 de las 2 de "T4 x2"). Tiempo total 214 min: **134 min descargando CheXmask desde PhysioNet**, 7 min de
preparación, 31 + 31 + 0,6 min de entrenamiento. Semilla 0 (una sola: aún no sirve para afirmar diferencias).

## Controles visuales

`control_chexmask.png` y `control_scr.png`: los contornos (débiles de CheXmask y humanos de SCR sobre JSRT limpia de
Kaggle) caen sobre pulmones y corazón en las 4 + 4 placas. Alineación JSRT ↔ `points.zip` **verificada a ojo**.

## E1 — segmentación, test dorado SCR (147 placas)

| Corrida | Dice pulmón der. | Dice pulmón izq. | Dice corazón | Dice medio | Error abs. ICT |
|---|---|---|---|---|---|
| E1a débil, todas | 0,9791 | 0,9686 | 0,9352 | 0,9609 | 0,0122 |
| E1b débil, RCA ≥ 0,7 | 0,9791 | 0,9689 | 0,9375 | 0,9618 | 0,0119 |
| E1c E1b + ajuste SCR | 0,9792 | 0,9708 | 0,9392 | 0,9631 | 0,0113 |

- Las diferencias son ≤ 0,004 de Dice con una semilla: **no se pueden declarar** hasta tener 3 semillas.
- Solo 110 de 8 776 máscaras fallan RCA ≥ 0,7, así que E1a y E1b entrenan casi con lo mismo.
- ⚠️ **E1c no es válido tal cual.** El mejor epoch se eligió con la validación débil de NIH, y ganó el epoch 1. Al
  ajustar a contornos humanos, el Dice contra CheXmask baja de 0,948 a 0,928. Ese dato también es un hallazgo: el
  estilo humano y el de CheXmask difieren. Corregido en `src/train_seg.py`: ahora el mejor epoch se elige con 20 de
  las 100 placas doradas. Hay que repetir E1c.

## Adelanto de E2/E4 (con `E1b_debil_rca07/ict_unet.csv`, test oficial NIH, n = 2 354, todas con ICT medible; en todo el subconjunto, 2 de 8 776 sin ICT)

| | AUC del ICT | Mediana ICT sin cardio | Mediana ICT con cardio |
|---|---|---|---|
| Todo el test | 0,814 | 0,497 | 0,563 |
| PA (n = 1 019) | 0,859 | 0,476 | 0,556 |
| AP (n = 1 335) | 0,779 | 0,512 | 0,569 |

- Con umbral 0,50: sensibilidad 0,916, especificidad 0,505. En AP, la mediana de los **normales** ya pasa 0,50: es
  la magnificación AP conocida en clínica. El atajo AP afecta también a la medición, no solo a la caja negra.
- Escala en cm: mediana MRD 4,7 cm, diámetro torácico 29,8 cm. Es plausible para adultos, pero falta validarlo contra
  una referencia.
- Calculado en local el 2026-11-02 con pandas + `roc_auc_score` sobre los CSV de esta carpeta.

## Pesos

Los `best.pt` están en `proyecto-final/modelos/kaggle-v1/*-best.zip`, fuera de git. Son el `.pt` tal cual (el formato
de torch ya es zip): `torch.load` los abre sin descomprimir.
