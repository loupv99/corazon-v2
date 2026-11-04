# Artículo IEEE

`main.tex` + `refs.bib` + `figs/`. Compilar (MiKTeX/TeX Live, sin perl):
`pdflatex main && bibtex main && pdflatex main && pdflatex main`. Con perl: `latexmk -pdf main.tex`.

**Estado 2026-11-04:** borrador de 5 páginas (límite 12). Metodología y Resultados completos con números verificados
contra `results/` (cada tabla lleva en un comentario el archivo del que sale). En rojo, lo pendiente:
- Introducción: motivación peruana con cifras **verificadas en su fuente** (hoy POR CONFIRMAR en
  `CONTEXTO-Y-ANTECEDENTES.md`).
- Discusión y Conclusiones: el viernes (los puntos ya están listados).
- Análisis fino de Grad-CAM.
- Contribuciones por integrante y declaración de uso de IA: las define el equipo.
- Enlace a los pesos.
- Título: provisional el candidato 1; se elige al final entre los 3 (comentario al inicio de `main.tex`).

**Referencias:** escritas de memoria del campo; antes de entregar, comprobar volumen, páginas y año de cada una en su
fuente (en especial Sogancioglu 2020, Chamveha 2020, CheXphoto 2020 y CheXmask 2024). Sogancioglu se cita sin sus
cifras porque no se leyeron en el artículo original.
