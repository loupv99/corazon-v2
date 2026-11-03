# Kaggle v2 — semillas, ResNet-50, Grad-CAM (notebook 02, corrido 2026-11-02/03)

GPU: 2 × Tesla T4, dos colas en paralelo. Tiempo total: 151 min, sin fallas. Minutos por tarea en
[tiempos_gpu.csv](tiempos_gpu.csv): U-Net 150,6 min (cola 0), ResNet-50 + Grad-CAM 38,2 min (cola 1).
Test oficial NIH: 2 354 placas (1 019 PA, 1 335 AP). Test dorado SCR: 147 placas. Media ± DE de 3 semillas.
Los pesos (`best.pt`, ~840 MB) no están en git: quedaron en `Downloads/results/runs/` de Lu.

## E1 — segmentación (test dorado SCR)

| Variante | Dice corazón | Dice medio | Error abs. ICT |
|---|---|---|---|
| E1a débil, todas las máscaras | 0,9358 ± 0,0008 | 0,9615 ± 0,0005 | 0,0119 ± 0,0003 |
| E1b débil, RCA ≥ 0,7 | 0,9360 ± 0,0014 | 0,9614 ± 0,0005 | 0,0121 ± 0,0002 |
| E1c E1b + ajuste con contornos humanos | 0,9512 ± 0,0006 | 0,9705 ± 0,0001 | 0,0099 ± 0,0006 |

Filtrar CheXmask por RCA no cambió nada medible. El ajuste con contornos humanos (E1c: 80 placas SCR para entrenar
y 20 para elegir la época; la mejor fue la 10, ya no la 1) mejora todo.

## E3 — acuerdo del ICT con contornos humanos (SCR, n = 147)

ICC(A,1): E1a 0,948–0,953 · E1b 0,945–0,951 · E1c 0,964–0,971. Sesgo E1c −0,0005 a −0,0035; límites de
Bland-Altman ≈ ±0,03. Acuerdo en la decisión ICT ≥ 0,50: 93–96 %. Detalle por corrida en
[E3_acuerdo_ict.csv](E3_acuerdo_ict.csv).

## E2 — diagnóstico de cardiomegalia (test NIH)

| Método | AUC | Sens. (Youden) | Esp. (Youden) |
|---|---|---|---|
| ResNet-50 PA + AP | 0,892 ± 0,010 | 0,830 | 0,786 |
| ResNet-50 solo PA | 0,865 ± 0,014 | 0,771 | 0,783 |
| ICT U-Net (E1b) | 0,814 ± 0,001 | 0,925 | 0,488 |
| ICT + vista (regresión logística) | 0,810 ± 0,001 | 0,834 | 0,615 |
| ICT CheXmask (sin red) | 0,801 | 0,930 | 0,471 |
| ICT U-Net ajustada (E1c) | 0,796 ± 0,006 | 0,955 | 0,407 |

ICT con el umbral clínico 0,50: sensibilidad 0,918, especificidad 0,504.

## E4 — proyección AP

| Método | AUC PA | AUC AP | Esp. AP (Youden) | Fuga de vista |
|---|---|---|---|---|
| ResNet-50 PA + AP | 0,944 | 0,845 | 0,750 | 0,646 ± 0,055 |
| ResNet-50 solo PA | 0,936 | 0,795 | 0,731 | 0,710 ± 0,045 |
| ICT U-Net | 0,864 | 0,776 | 0,356 | 0,691 ± 0,011 |
| ICT + vista | 0,864 | 0,776 | 0,619 | 0,493 ± 0,011 |

Fuga de vista = AUC del puntaje para separar AP de PA entre placas sin cardiomegalia (0,5 = no codifica la vista).

## Lectura (pendiente de redactar en el paper)

1. La U-Net mide el ICT con acuerdo alto frente a humanos (ICC 0,97 con ajuste), entrenada casi solo con máscaras
   automáticas.
2. Como clasificador, la ResNet-50 supera al ICT en AUC (0,89 frente a 0,81). La hipótesis "detecta tan bien como la
   caja negra" NO se sostiene.
3. Medir mejor (E1c) no diagnosticó mejor (AUC 0,796 < 0,814). Explicación probable, POR CONFIRMAR: las etiquetas NIH
   vienen de informes y el umbral 0,50 no es la definición que usó cada radiólogo.
4. Ambos métodos pierden en AP. En el ICT la causa es conocida (magnificación) y se corrige con la vista: la fuga baja
   a 0,49 y la especificidad en AP sube de 0,36 a 0,62. La ResNet entrenada solo con PA pierde 0,14 de AUC en AP.
5. Grad-CAM ([E4_gradcam.csv](E4_gradcam.csv)): cuando la ResNet dice "positivo", la energía en el corazón es 4–6
   veces su área, acierte o no; en los negativos correctos, 1,2–4. Falta el análisis fino.
