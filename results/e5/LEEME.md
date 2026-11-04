# E5 — radiografías fotografiadas con celular (sesión 2026-11-03/04)

100 placas del test oficial (25 por grupo: PA/AP × con/sin cardiomegalia), mostradas en un monitor de 1920×1080 con
4 marcas ArUco de esquina y 1 de número (`src/fotos.py`). Análisis en CPU con `src/e5.py`: 3 semillas de la U-Net
E1b (ICT, umbral 0,50) y de la ResNet-50 PA + AP (umbral de Youden de su validación), 35,5 min.

| Celular | Toma | Fotos leídas | Placas | Marcas de esquina vistas (4 / 3 / 2) |
|---|---|---|---|---|
| iPhone 16 Pro | de frente | 100/100 | 100 | 81 / 15 / 4 |
| iPhone 16 Pro | inclinada | 100/100 | 100 | 81 / 18 / 1 |
| Honor (modelo POR CONFIRMAR) | de frente | 101/101 | 100 | 74 / 25 / 2 |

En el Honor la placa 64 se fotografió dos veces; vale la última. Las fotos del iPhone se tomaron con ProRAW activo y
se usó el JPG que el teléfono genera; la asignación foto → placa y frente/inclinada está en `asignacion_iphone.csv`
(se separaron por la razón entre el lado mayor y el menor de las marcas: la foto más pareja de cada placa es "frente").
**La inclinación real fue moderada** (razón de lados 1,13–1,32 en las inclinadas y 1,00–1,15 en las de frente), no los
30° previstos.

## Resultado (media ± DE de 3 semillas, 100 placas; `E5_resumen.csv`)

| Condición | ICT: AUC | ICT: cambia la decisión | ResNet: AUC | ResNet: cambia la decisión |
|---|---|---|---|---|
| original | 0,827 ± 0,008 | — | 0,899 ± 0,020 | — |
| iPhone de frente | 0,801 ± 0,009 | 6,7 % ± 4,7 | 0,872 ± 0,020 | 13,3 % ± 4,7 |
| iPhone inclinada | 0,797 ± 0,014 | 8,3 % ± 3,5 | 0,883 ± 0,010 | 10,3 % ± 2,1 |
| Honor de frente | 0,812 ± 0,014 | 4,7 % ± 2,1 | 0,881 ± 0,011 | 8,3 % ± 1,5 |
| sintéticas (rango de 6) | 0,814–0,833 | 2,0–5,3 % | 0,891–0,908 | 4,3–22,3 % |

- |ΔICT| mediana entre foto y original: 0,010–0,012. Sesgo medio cercano a 0 (−0,006 a +0,001).
- Las placas cuyo ICT cambia de lado del umbral están cerca de él: distancia mediana al 0,50 de **0,013** en esas
  59 de 899 mediciones, frente a 0,056 en todas (3 tomas × 3 semillas × ~100 placas).
- 1 placa no medible (semilla 2, iPhone de frente, k = 70) de 900 mediciones de ICT con foto.

## Corregida vs. sin corregir, mismas placas (`E5_pareado_sin_corregir.csv`)

"Sin corregir" solo existe cuando se vieron las 4 marcas (73–81 placas). Sobre esas mismas placas la corrección de
perspectiva cambia poco: ICT 5,9 vs 8,2 % (Honor), 6,2 vs 7,0 % (iPhone frente), 10,3 vs 10,7 % (iPhone inclinada) de
decisiones que cambian; ResNet 7,3 vs 8,7 %, 11,9 vs 9,5 %, 11,5 vs 15,2 %. Con inclinaciones moderadas, la
homografía no es la que sostiene la robustez.

## Lectura

1. Ninguno de los dos métodos se desarma con la foto: ambos pierden unos 0,02–0,03 de AUC.
2. El ICT cambia de decisión en menos placas que la ResNet (4,7–8,3 % frente a 8,3–13,3 %), y cuando cambia es en casos
   limítrofes. Con una sola sesión y 100 placas, si la diferencia es significativa está **POR CONFIRMAR**.
3. Las degradaciones sintéticas subestiman el efecto real sobre el ICT (2–5 % frente a 5–8 % de decisiones que
   cambian).
4. Las diferencias de |Δ| entre modelos no se comparan directamente: el ICT y la probabilidad de la ResNet están en
   escalas distintas. La comparación justa es AUC y % de decisiones que cambian.

Las fotos (~2,8 GB) no se suben al repositorio.
