# App web — medir el ICT en el navegador

Una sola página (`index.html`) que corre la U-Net en el dispositivo con ONNX Runtime Web 1.20.1: la imagen no se sube
a ningún servidor. Interfaz según el diseño de Figma del 2026-11-06 (visor tipo PACS, IBM Plex Sans/Mono, tema oscuro
y claro).

1. **Inicio.** Subir una radiografía (o arrastrarla), tomar una foto o abrir un ejemplo (`?ejemplo=con` /
   `?ejemplo=sin` lo abre directo). Datos opcionales del estudio (historia clínica, edad, sexo, fecha, proyección
   PA/AP): quedan solo en la página, no se guardan ni se envían; la proyección se anota en el informe y, si es AP,
   se advierte que el corazón se ve más grande.
2. **Marcar esquinas** (si es una foto de un negatoscopio o un monitor): se tocan las 4 esquinas en orden, con lupa
   ×3 y vista previa enderezada; la homografía es la de `src/fotos.py` (900 px antes de reducir).
3. **Medición.** Contornos (pulmones en cian, corazón en ámbar), fila del diámetro torácico (ID), línea media y bordes
   del corazón (MRD, MLD), en píxeles de la imagen de 256 px. «Corregir bordes» muestra 4 asas que se arrastran; el
   ICT se recalcula al momento, se muestra la diferencia con la automática y se puede volver a ella.
   «Copiar al informe» copia una línea con el ICT, MRD, MLD, ID, la proyección y si la medida se corrigió.
4. **Carga del modelo** con progreso real de la descarga (31 MB, solo la primera vez) y **no medible** cuando faltan
   los pulmones o el corazón.

**Detector de baja calidad** (2026-11-06; `src/calidad.py`, `results/calidad/LEEME.md`). Sobre la imagen de 256 px y
la salida de la red calcula contraste, nitidez, exposición, incertidumbre de la red y estructuras cortadas por el
borde, con umbrales = percentil 1 o 99 de las 100 placas originales de E5. **Solo avisan** la incertidumbre, las
estructuras cortadas y la inclinación de la foto (razón de lados de las esquinas > 1,15, el criterio de E5): son los
que señalaron imágenes con más error; contraste, nitidez y exposición no lo predijeron y solo se calculan. Los cinco
indicadores coinciden con Python en las dos placas de ejemplo (diferencias en el tercer decimal por el redondeo de la
reducción a 256 px).

**Mismo cálculo que Python** (verificado 2026-11-04 y de nuevo con la interfaz nueva el 2026-11-06): gris como
`cv2.COLOR_BGR2GRAY`, reducción por área a 256 px como `cv2.INTER_AREA`, normalización (g/255 − 0,5)/0,25, umbral
0,5, mayor componente conexa y `ctr.medir`.
- Ejemplos: ICT 0,5219 y 0,4456, con los mismos píxeles de borde que `src.ctr` (ID 228 y 193 px).
- Foto real del iPhone (inclinada, placa 1) con las esquinas exactas: ICT 0,46591 en la app y en Python (2026-11-04).
- ONNX frente a PyTorch: diferencia máxima de probabilidad 2,5·10⁻⁶; 100 % de píxeles iguales
  (`modelo/unet_verificacion.json`).

**Modelo:** `modelo/unet.onnx` (31 MB) = U-Net E1b semilla 1, elegida por Dice de validación (0,9596), no por el test.
Se regenera con `python -m src.exportar_onnx --ckpt runs/E1b_debil_rca07_s1/best.pt --out app/modelo/unet.onnx`.

**Probar en local:** `cd app && python -m http.server 8000` y abrir http://localhost:8000 (como archivo suelto no
carga el modelo). La primera vez necesita conexión para ONNX Runtime y las fuentes.

**Límites:** la app no detecta las marcas ArUco (haría falta OpenCV.js, ~8 MB): las esquinas se marcan a mano. No
reporta cm porque una foto no trae el tamaño del píxel. El detector de calidad se calibró con 100 placas y 300 fotos
de una sola sesión: es una alerta, no una garantía. Es una herramienta académica, no un dispositivo médico.
