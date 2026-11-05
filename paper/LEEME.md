# Artículo IEEE

`main.tex` + `refs.bib` + `figs/`. Compilar (MiKTeX/TeX Live, sin perl):
`pdflatex main && bibtex main && pdflatex main && pdflatex main`. Con perl: `latexmk -pdf main.tex`.

**Estado 2026-11-04 (tarde):** 6 páginas (límite 12). Metodología, Resultados (con el análisis de Grad-CAM por
acierto y error, de `results/kaggle-v2/E4_gradcam.csv`), Discusión y Conclusiones redactados con números verificados
contra `results/`. En rojo, lo pendiente:
- Introducción: motivación peruana con cifras **verificadas en su fuente** (hoy POR CONFIRMAR en
  `CONTEXTO-Y-ANTECEDENTES.md`).
- Contribuciones por integrante y declaración de uso de IA: las define el equipo. **Ambas son obligatorias** según el
  enunciado (sin Contribuciones, el artículo se considera incompleto en el criterio 6).
- Enlace a los pesos.
- Título: provisional el candidato 1; se elige al final entre los 3 (comentario al inicio de `main.tex`).

**Referencias:** escritas de memoria del campo; antes de entregar, comprobar volumen, páginas y año de cada una en su
fuente (en especial Sogancioglu 2020, Chamveha 2020, CheXphoto 2020 y CheXmask 2024). Sogancioglu se cita sin sus
cifras porque no se leyeron en el artículo original.

**Verificadas en la fuente (2026-11-04):**
- CheXmask usa HybridGNet. Cita literal: "Our methodology utilizes the HybridGNet model to ensure consistent and
  high-quality segmentations across all datasets." — https://arxiv.org/abs/2307.03293
- HybridGNet = codificador convolucional + decodificador de grafos. Cita literal: "an encoder-decoder neural
  architecture that leverages standard convolutions for image feature encoding"; usa redes convolucionales de grafos
  para "decode plausible representations of anatomical structures". Aceptado en IEEE TMI. —
  https://arxiv.org/abs/2203.10977
