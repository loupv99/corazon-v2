# Guion de la sustentación (25 min + 5 de preguntas)

`slides.pdf` = 20 diapositivas. Compilar: `python slides/figuras.py` (desde la raíz) y luego, en `slides/`,
`pdflatex slides` dos veces. Las cifras salen de `results/` (las mismas del artículo).

La columna **Expone** la llena el equipo: todos exponen y el jurado puede preguntar a cualquiera sobre cualquier parte.

| # | Diapositiva | Bloque (min) | Seg. | Expone | Idea que debe quedar |
|---|---|---|---|---|---|
| 1 | Portada | Problema (3) | 15 | | Equipo y pregunta en una frase |
| 2 | El problema | | 75 | | El ICT es una medida; sin radiólogo, la placa espera |
| 3 | Pregunta y contribuciones | | 90 | | Medir vs. clasificar; 3 contribuciones |
| 4 | Trabajos relacionados | Relacionados (3) | 90 | | Qué hace cada uno y en qué nos diferenciamos |
| 5 | Fundamento | | 90 | | Saltos de la U-Net = bordes; residuales; fórmula de Grad-CAM |
| 6 | Datos | Metodología (6) | 70 | | Partición por paciente; la vista está asociada a la etiqueta |
| 7 | Etiquetas débiles, referencia humana | | 50 | | Aprende de HybridGNet, se evalúa contra personas |
| 8 | Arquitectura | | 60 | | Dos caminos para la misma placa |
| 9 | Formulación | | 70 | | BCE + Dice por canal; del contorno al ICT |
| 10 | Entrenamiento y costo | | 50 | | 3 semillas, sin fuga, 4.2 h de GPU |
| 11 | Diseño experimental | | 60 | | E1–E6 y la definición de fuga de vista |
| 12 | Resultado 1 | Resultados (7) | 80 | | ResNet 0.892 vs. ICT 0.814; el umbral no es el problema |
| 13 | Resultado 2 | | 80 | | ICC 0.945–0.951 sin humanos; paradoja de E1c |
| 14 | Resultado 3 | | 90 | | Todos pierden en AP; el ICT se corrige con la vista |
| 15 | Resultado 4 | | 90 | | El ICT cambia menos y solo cerca de 0.50 |
| 16 | La aplicación | | 80 | | ONNX = PyTorch; el médico arrastra la línea |
| 17 | Grad-CAM | Discusión (4) | 70 | | El mapa se ve igual cuando acierta y cuando falla |
| 18 | Discusión | | 80 | | Explicar no es medir; E6: el ICT no ve el ventrículo derecho |
| 19 | Limitaciones e impacto | | 90 | | Apoyo, no diagnóstico; la imagen no sale del teléfono |
| 20 | Conclusiones | Conclusiones (2) | 120 | | Qué error puede revisar quien usa la herramienta |

Total: 1 500 s = 25 min. Ensayar con cronómetro; si se usa la app en vivo, sale del tiempo de la diapositiva 16.

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
