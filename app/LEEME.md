# App web — medir el ICT en el navegador

Una sola página (`index.html`) que corre la U-Net en el dispositivo con ONNX Runtime Web 1.20.1: la imagen no se sube
a ningún servidor.

1. Subir una radiografía, tomar una foto o abrir un ejemplo (`?ejemplo=con` / `?ejemplo=sin` lo abre directo).
2. Si es una foto de una pantalla o de una placa: marcar las 4 esquinas de la radiografía; se endereza con una
   homografía (como `src/fotos.py`, a 900 px) antes de reducirla.
3. Se ven los contornos (pulmones en azul, corazón en rojo), la fila del diámetro torácico, la línea media y los bordes
   del corazón. Las 4 líneas verticales se pueden arrastrar para corregir la medida; el ICT se recalcula al momento.

**Mismo cálculo que Python** (verificado 2026-11-04): gris como `cv2.COLOR_BGR2GRAY`, reducción por área a 256 px como
`cv2.INTER_AREA`, normalización (g/255 − 0,5)/0,25, umbral 0,5, mayor componente conexa y `ctr.medir`.
- Ejemplos: ICT 0,5219 y 0,4456, con los mismos píxeles de borde que `src.ctr` (ID 228 y 193 px).
- Foto real del iPhone (inclinada, placa 1) con las esquinas exactas: ICT 0,46591 en la app y en Python.
- ONNX frente a PyTorch: diferencia máxima de probabilidad 2,5·10⁻⁶; 100 % de píxeles iguales
  (`modelo/unet_verificacion.json`).

**Modelo:** `modelo/unet.onnx` (31 MB) = U-Net E1b semilla 1, elegida por Dice de validación (0,9596), no por el test.
Se regenera con `python -m src.exportar_onnx --ckpt runs/E1b_debil_rca07_s1/best.pt --out app/modelo/unet.onnx`.

**Probar en local:** `cd app && python -m http.server 8000` y abrir http://localhost:8000 (como archivo suelto no
carga el modelo).

**Límites:** la app no detecta las marcas ArUco (haría falta OpenCV.js, ~8 MB): las esquinas se marcan a mano. No
reporta cm porque una foto no trae el tamaño del píxel. Es una herramienta académica, no un dispositivo médico.
