# Artículo IEEE

`main.tex` + `refs.bib` + `figs/`. Compilar (MiKTeX/TeX Live, sin perl):
`pdflatex main && bibtex main && pdflatex main && pdflatex main`. Con perl: `latexmk -pdf main.tex`.

**Estado 2026-11-04 (tarde):** 6 páginas (límite 12). Metodología, Resultados (con el análisis de Grad-CAM por
acierto y error, de `results/kaggle-v2/E4_gradcam.csv`), Discusión y Conclusiones redactados con números verificados
contra `results/`. La motivación peruana (Introducción) usa dos cifras verificadas en su fuente (radiólogos y
Telesalud; citas literales en `../CONTEXTO-Y-ANTECEDENTES.md`, §3). No queda texto en rojo. Quedan **en blanco**, para
el equipo:
- La tabla de Contribuciones y la declaración de uso de IA. **Ambas son obligatorias** según el enunciado (sin
  Contribuciones, el artículo se considera incompleto en el criterio 6).

Enlace a los pesos: completado el 2026-11-07, en la sección «Disponibilidad de código». Lleva al Release
`v1.0-pesos` (15 `.pt` + `SHA256SUMS.txt`).
- Título: provisional el candidato 1; se elige al final entre los 3 (comentario al inicio de `main.tex`).

**Referencias comprobadas (2026-11-06):** las 17 académicas, contra Crossref (DOI), arXiv, PMLR y OpenAlex: título,
autores, revista, volumen, páginas y año. Correcciones hechas:
- ChestX-ray8: páginas 3462--3471 (las de IEEE Xplore, DOI 10.1109/CVPR.2017.369; antes 2097--2106, que es la
  paginación de la versión abierta de CVF).
- HybridGNet: ya publicado, IEEE TMI 42(2):546--556, 2023, DOI 10.1109/TMI.2022.3224660 (antes como arXiv).
- CheXphoto: publicado en ML4H, PMLR 136:318--327, 2020 (https://proceedings.mlr.press/v136/phillips20a.html).
- CheXmask: falta el número 1 (añadido).
- U-Net (LNCS 9351, 234--241, DOI 10.1007/978-3-319-24574-4_28), He, Grad-CAM, SCR, JSRT, Sogancioglu, Zech,
  McGraw, Bland-Altman, ArUco, Youden y AdamW (arXiv 1711.05101: "Published as a conference paper at ICLR 2019"):
  sin cambios.

**Afirmaciones sobre otros trabajos, comprobadas en la fuente (2026-11-06):**
- Sogancioglu 2020 (resumen, DOI 10.1109/ACCESS.2020.2995567): "obtained heart and lung segmentation annotations
  for 778 chest radiographs"; prueba en "367 held-out images for which cardiomegaly annotations were hand-labeled by
  two independent expert radiologists"; "segmentation-based system with an AUC of 0.977 is significantly better ...
  than the classification-based model which achieved an AUC of 0.941". Sus cifras ya se citan en el paper.
- Chamveha 2020 (https://arxiv.org/abs/2002.07468): no propusieron la descomposición; la toman de sus refs. [2, 3]:
  "CTR can be calculated from three measurements, MRD, ... MLD, ... and ID ... [2, 3]". Resumen: "76.5% were
  accepted to be included in medical reports without any need for adjustment". Redacción corregida.
- CheXphoto (https://arxiv.org/abs/2007.06199): es un conjunto de datos "for testing and improving the robustness";
  la frase del paper ya no dice que midió pérdidas.
- JSRT (Shiraishi 2000, resumen en OpenAlex, DOI 10.2214/ajr.174.1.1740071): radiografías convencionales "from 14
  medical centers and were digitized by a laser digitizer with a 2048 x 2048 matrix size". PA, en fuente secundaria:
  "This public database includes 247 posterior-anterior (PA) chest radiograph images" —
  https://arxiv.org/abs/1810.02113 (artículo de 2018 sobre U-Net con codificador preentrenado).

**Verificadas en la fuente (2026-11-04):**
- CheXmask usa HybridGNet. Cita literal: "Our methodology utilizes the HybridGNet model to ensure consistent and
  high-quality segmentations across all datasets." — https://arxiv.org/abs/2307.03293
- HybridGNet = codificador convolucional + decodificador de grafos. Cita literal: "an encoder-decoder neural
  architecture that leverages standard convolutions for image feature encoding"; usa redes convolucionales de grafos
  para "decode plausible representations of anatomical structures". Aceptado en IEEE TMI. —
  https://arxiv.org/abs/2203.10977
