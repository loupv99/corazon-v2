# Guion de la sustentación (25 min + 5 de preguntas)

`slides.pdf` = 23 diapositivas (el enunciado *sugiere* 15–20; lo estricto son los 25 min), con el formato
institucional UNI de `formato-diapo.pptx` (escudo, granate, Calibri/Cambria vía Carlito/Caladea; logo en
`figs/logo_uni.png`). Compilar: `python slides/figuras.py` (desde la raíz) y luego, en `slides/`,
`pdflatex slides` dos veces. Las cifras salen de `results/` (las mismas del artículo).

La columna **Expone** la llena el equipo: todos exponen y el jurado puede preguntar a cualquiera sobre cualquier parte.

| # | Diapositiva | Bloque (min) | Seg. | Expone | Idea que debe quedar |
|---|---|---|---|---|---|
| 1 | Portada | Problema (3) | 15 | | Equipo y pregunta en una frase |
| 2 | Contexto y antecedentes | | 55 | | De Danzer (1919) a CheXmask (2024); en el Perú la placa viaja, falta quien la lea |
| 3 | El problema | | 50 | | El ICT es una medida; sin radiólogo, la placa espera |
| 4 | Pregunta, hipótesis y contribuciones | | 60 | | Medir vs. clasificar; hipótesis; 3 contribuciones y su experimento |
| 5 | Trabajos relacionados (1/2) | Relacionados (3) | 60 | | Arquitecturas y datos: metodología, hallazgo, aporte, diferencia |
| 6 | Trabajos relacionados (2/2) | | 60 | | Sogancioglu ganó midiendo con etiquetas de radiólogos; aquí no |
| 7 | Fundamento | | 60 | | Saltos de la U-Net = bordes; bloque residual; fórmula de Grad-CAM |
| 8 | Preparación de datos | Metodología (6) | 55 | | Cuatro fuentes, un mismo formato; JSRT limpias, partición por paciente |
| 9 | Datos | | 50 | | La vista está asociada a la etiqueta; qué papel cumple cada fuente |
| 10 | Etiquetas débiles, referencia humana | | 45 | | Maestro HybridGNet → alumna U-Net → juez humano |
| 11 | Arquitectura | | 50 | | Dos caminos para la misma placa |
| 12 | Formulación | | 55 | | BCE + Dice por canal; del contorno al ICT (MRD, MLD, ID) |
| 13 | Entrenamiento y costo | | 50 | | 5 configuraciones × 3 semillas; la prueba no elige nada; 4.2 h de GPU |
| 14 | Diseño experimental | | 55 | | E1–E6 y la definición de fuga de vista |
| 15 | Resultado 1 | Resultados (7) | 80 | | ResNet 0.892 vs. ICT 0.814; el umbral no es el problema |
| 16 | Resultado 2 | | 80 | | ICC 0.945–0.951 sin humanos; paradoja de E1c |
| 17 | Resultado 3 | | 90 | | Todos pierden en AP; el ICT se corrige con la vista |
| 18 | Resultado 4 | | 90 | | El ICT cambia menos y solo cerca de 0.50 |
| 19 | La aplicación | | 80 | | ONNX = PyTorch; el médico corrige los bordes; aviso de calidad (20 % vs. 2,9 %) |
| 20 | Grad-CAM | Discusión (4) | 70 | | El mapa se ve igual cuando acierta y cuando falla |
| 21 | Discusión | | 80 | | Explicar no es medir; E6: el ICT no ve el ventrículo derecho |
| 22 | Limitaciones e impacto | | 90 | | Apoyo, no diagnóstico; la imagen no sale del teléfono |
| 23 | Conclusiones | Conclusiones (2) | 120 | | Qué error puede revisar quien usa la herramienta |

Total: 1 500 s = 25 min (bloques 180 / 180 / 360 / 420 / 240 / 120 s). Ensayar con cronómetro; si se usa la app en
vivo, sale del tiempo de la diapositiva 19.

## Preguntas probables (con el dato para responder)

- **¿Por qué no ajustaron el umbral del ICT?** Se dejó el clínico de 0.50; el de Youden en validación da 0.495–0.497,
  casi igual (`results/kaggle-v2/E2_E4_por_semilla.csv`).
- **¿Hay fuga entre particiones?** Partición por *Patient ID*; la prueba es la lista oficial del NIH; nada se eligió
  con la prueba.
- **¿Por qué E1c diagnostica peor si mide mejor?** La etiqueta del NIH no es "ICT > 0.50", y la mejora se midió en
  JSRT, el mismo dominio del ajuste; no aislamos cuál pesa más.
- **Sogancioglu (2020) encontró que la segmentación gana (AUC 0.977 frente a 0.941); ¿por qué ustedes no?** Ellos
  probaron con 367 placas etiquetadas por dos radiólogos y entrenaron con 778 placas anotadas; nosotros, con la
  etiqueta del informe del NIH y máscaras automáticas. Las dos diferencias pueden invertir el resultado; no las
  aislamos (cifras de su resumen, DOI 10.1109/ACCESS.2020.2995567).
- **¿Qué se reutilizó?** ResNet-50 de ImageNet (ajustada completa), máscaras de CheXmask; la U-Net se entrenó desde
  cero (código del examen parcial del grupo).
- **¿Por qué no Transformers?** Se plantea como trabajo futuro; la fuente de máscaras (HybridGNet) ya es una red de
  grafos reciente que destilamos en una U-Net.
