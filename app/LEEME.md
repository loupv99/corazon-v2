# App web — medir el ICT en el navegador

Una sola página (`index.html`) que corre la U-Net en el dispositivo con ONNX Runtime Web 1.20.1: la imagen no se sube
a ningún servidor. Interfaz según el diseño de Figma del 2026-11-06 (visor tipo PACS, IBM Plex Sans/Mono, tema oscuro
y claro).

1. **Inicio.** Subir una radiografía PNG, JPG o DICOM (o arrastrarla), tomar una foto o abrir un ejemplo
   (`?ejemplo=con`, `?ejemplo=sin` o `?ejemplo=dicom` lo abre directo). Datos opcionales del estudio (historia clínica, edad, sexo, fecha, proyección
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

**DICOM** (2026-11-06). Abre DICOM sin compresión (Explicit/Implicit VR Little Endian) con dicom-parser 1.8.21:
8 bits se usan tal cual (mismo ICT que el PNG); 12-16 bits con la ventana del archivo (o mínimo-máximo) y
MONOCHROME1 invertido. Si trae **PixelSpacing**, MRD, MLD e ID salen también en **cm** (mm por píxel a 256 px =
espaciado de columna × ancho / 256, como `src/prepare_data.py`); si solo trae ImagerPixelSpacing, lo usa y el informe
avisa que está en el detector, sin corregir magnificación. Completa la proyección (ViewPosition), la edad, el sexo y
la fecha si están vacíos; no lee el nombre del paciente. Un DICOM comprimido (JPEG, JPEG 2000, RLE) da un aviso y no
se mide. `ejemplos/*.dcm` = las dos placas de ejemplo convertidas con `python -m src.dicom` (captura secundaria: misma
imagen de 8 bits del NIH y su tamaño de píxel; **no** son los estudios originales; el espaciado vertical repite el
horizontal porque el NIH estiró las placas a 1 024 × 1 024 y solo se usan anchos). Verificado en Chrome el 2026-11-06:
ejemplo «con» 0,52 con MRD 4,9 · MLD 13,3 · ID 35,0 cm (tabla de E1b semilla 1: 4,91 · 13,35 · 34,98); «sin» 0,45
con 4,1 · 10,3 · 32,3 cm; un DICOM de prueba de 16 bits MONOCHROME1 dio 0,52 (MLD 86 px en vez de 87 por la ventana).
Para subir en la demo hay además dos placas de la selección de E5 (2026-11-07,
`python -m src.dicom --sel e5_seleccion --imagenes 00028509_016.png 00028445_000.png`):
`ejemplos/demo_AP_con_cardiomegalia.dcm` (00028509_016, AP: la app pone AP y avisa de la magnificación; ICT 0,63,
MRD 8,5 · MLD 11,4 · ID 31,7 cm) y `ejemplos/demo_PA_sin_cardiomegalia.dcm` (00028445_000, PA; ICT 0,38,
2,9 · 7,3 · 26,8 cm). Con `--imagenes 00023033_014.png 00014635_004.png` se completan las cuatro combinaciones:
`ejemplos/demo_AP_sin_cardiomegalia.dcm` (00023033_014, AP; ICT 0,44, 3,6 · 10,2 · 31,4 cm) y
`ejemplos/demo_PA_con_cardiomegalia.dcm` (00014635_004, PA; ICT 0,65, 8,0 · 12,6 · 31,8 cm). Las cuatro verificadas
en Chrome; los cm coinciden con la tabla de E1b semilla 1.

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
El `best.pt` está publicado como `E1b_debil_rca07_s1.pt` en el Release
https://github.com/loupv99/corazon-v2/releases/tag/v1.0-pesos (verificado 2026-11-07: con la misma entrada, la
sigmoide de PyTorch y el ONNX difieren como máximo 1,3·10⁻⁷).

**En línea:** https://loupv99.github.io/corazon-v2/ (GitHub Pages, rama `gh-pages` = el contenido de `app/` en la raíz
más `.nojekyll`; verificada el 2026-11-06: el ejemplo «con» da 0,52). Si cambia `app/`, hay que actualizar
`gh-pages`; al subirla, usar `git -c pack.useSparse=false push`, o git vuelve a enviar el modelo de 31 MB.

**Probar en local:** `cd app && python -m http.server 8000` y abrir http://localhost:8000 (como archivo suelto no
carga el modelo). La primera vez necesita conexión para ONNX Runtime y las fuentes.

**Límites:** la app no detecta las marcas ArUco (haría falta OpenCV.js, ~8 MB): las esquinas se marcan a mano. Con PNG,
JPG o fotos no reporta cm porque no traen el tamaño del píxel (solo los DICOM que lo traen). No abre DICOM
comprimidos. El detector de calidad se calibró con 100 placas y 300 fotos
de una sola sesión: es una alerta, no una garantía. Es una herramienta académica, no un dispositivo médico.
