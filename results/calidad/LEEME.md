# Detector de baja calidad de la app (2026-11-06)

`python -m src.calidad --sel e5_seleccion --fotos fotos --onnx app/modelo/unet.onnx --out results/calidad`
(CPU, el mismo modelo ONNX de la app). Imágenes: las 100 placas originales de E5, sus 300 fotos corregidas con las
marcas ArUco (iPhone 16 Pro de frente e inclinada, Honor X8b de frente), 235 de esas fotos sin corregir y las 600
degradaciones sintéticas (6 × 100).

**Indicadores** (sobre la imagen de 256 px y la salida de la red, iguales en `app/index.html`): contraste (DE del gris),
nitidez (varianza del laplaciano), exposición (fracción de píxeles ≤ 5 o ≥ 250), incertidumbre (píxeles con
probabilidad entre 0,2 y 0,8 / píxeles ≥ 0,5) y borde (fracción de pulmón o corazón en el marco).
**Umbral** = percentil 1 (contraste, nitidez) o 99 (los demás) de las 100 originales (`umbrales.json`).

## Qué indicador predice error (`calidad_resumen.csv`; |ΔICT| y cambio de decisión frente a la placa original)

| Criterio | Fotos corregidas señaladas | \|ΔICT\| mediana señaladas vs. resto | Cambia la decisión señaladas vs. resto |
|---|---|---|---|
| **Incertidumbre** | 25 / 300 | **0,031 vs. 0,008** | **20 % vs. 2,9 %** |
| Borde | 0 / 300 (11 / 600 sintéticas) | — (sintéticas: 0,007 vs. 0,006) | — (sintéticas: 9,1 % vs. 2,4 %) |
| Contraste | 1 / 300 | 0,007 vs. 0,009 | 0 % vs. 4,3 % |
| Nitidez | 0 / 300 (122 / 600 sintéticas, 98 % del desenfoque) | — (sintéticas: 0,004 vs. 0,006) | — |
| Exposición | 0 / 300 | — | — |

Fotos sin corregir: incertidumbre señala 26 / 235, 0,034 vs. 0,010 y 19 % vs. 4,8 %. Sintéticas: 21 / 600,
0,020 vs. 0,006 y 14 % vs. 2,1 %.

**Decisión:** la app solo AVISA con incertidumbre y borde (más la inclinación de la foto, razón de lados > 1,15 como
en E5). Contraste, nitidez y exposición se calculan pero no avisan: no señalaron imágenes con más error (el
desenfoque, por ejemplo, se señala casi siempre y su error es menor), así que solo producirían falsas alarmas.

## Lo que el aviso logra y lo que no

- Fotos corregidas: el aviso sale en 25 de 300; de las 13 fotos cuyo ICT cambia de lado del 0,50, el aviso marca 5
  (38 %). Es una **alerta de riesgo**, no un filtro: la mayoría de los cambios de decisión ocurren en placas cerca del
  umbral (E5: mediana a 0,013 de 0,50) que la imagen no delata.
- Originales: 2 de 100 avisadas. Es una cifra dentro de la muestra, porque los umbrales salieron de esas mismas 100.
- Una sola sesión de fotos, dos celulares, un operador y 100 placas; las fotos se enderezaron con las esquinas exactas
  de ArUco, mientras que en la app las marca la persona a mano. No se probó con placas físicas en negatoscopio.
