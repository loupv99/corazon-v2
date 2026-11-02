"""Etiquetas doradas: contornos humanos de la base SCR (Zenodo 7056076, CC BY 4.0) sobre las radiografías JSRT.

Verificado el 2026-11-02:
  - points.zip  247 archivos .pfs con contornos por estructura en coordenadas de 1024 px; los 247 tienen
                "right lung", "left lung" y "heart". ESTO es la etiqueta dorada.
  - jpg.zip     247 JPG de 256 px, pero con los contornos DIBUJADOS encima (en 221/247 el borde del corazón es
                >15 niveles más oscuro que 2 px adentro). NO sirven como entrada del modelo: la red vería la
                respuesta. Las imágenes limpias se toman de otra copia de JSRT (p. ej. la PNG de Kaggle
                raddar/nodules-in-chest-xrays-jsrt, licencia POR CONFIRMAR) o del .IMG original del JSRT.
  - masks.zip   247 TIF con un solo pulmón: no se usa.
"""
from __future__ import annotations

import io
import os
import re
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

ESTRUCTURAS = ("right lung", "left lung", "heart")   # mismo orden de canales que chexmask.COLS
TAM_ORIGINAL = 1024

_BLOQUE = re.compile(r"\{\s*\[Label=([^\]]*)\](.*?)\n\}", re.S)
_PUNTO = re.compile(r"\{\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}")


def parse_pfs(texto: str) -> dict[str, np.ndarray]:
    """Archivo .pfs -> {etiqueta: array (N, 2) de puntos x, y}."""
    out = {}
    for label, cuerpo in _BLOQUE.findall(texto):
        pts = np.array([[float(x), float(y)] for x, y in _PUNTO.findall(cuerpo)])
        if len(pts):
            out[label.strip()] = pts
    return out


def rasterize(contornos: dict[str, np.ndarray], size: int) -> np.ndarray:
    """Contornos (coords de 1024 px) -> máscaras (3, size, size) uint8, una por estructura."""
    escala = size / TAM_ORIGINAL
    masks = np.zeros((len(ESTRUCTURAS), size, size), dtype=np.uint8)
    for k, nombre in enumerate(ESTRUCTURAS):
        lienzo = Image.new("L", (size, size), 0)
        ImageDraw.Draw(lienzo).polygon([tuple(p) for p in contornos[nombre] * escala], fill=1)
        masks[k] = np.array(lienzo)
    return masks


def leer_imagen(images_dir: str, sid: str) -> np.ndarray:
    """Busca la radiografía limpia por id (JPCLN001...) en PNG/JPG o en el .IMG original del JSRT."""
    for ext in (".png", ".PNG", ".jpg", ".IMG", ".img"):
        p = os.path.join(images_dir, sid + ext)
        if not os.path.exists(p):
            continue
        if ext.lower() == ".img":
            # formato original: 2048 x 2048, 12 bits, big-endian, sin cabecera
            a = np.fromfile(p, dtype=">u2").reshape(2048, 2048).astype(np.float32)
            return (a / a.max() * 255).astype(np.uint8)
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    raise FileNotFoundError(f"no se encontró la imagen de {sid} en {images_dir}")


def load_scr(points_zip: str, images_dir: str, size: int = 256):
    """Devuelve (ids, imágenes (N, size, size) uint8, máscaras (N, 3, size, size) uint8, invertidas: bool).

    Si en las imágenes los pulmones salen más claros que el corazón, la escala de grises está al revés (pasa con el
    .IMG del JSRT) y se invierte. La comprobación se hace con las máscaras doradas, no a ojo.
    """
    zp = zipfile.ZipFile(points_zip)
    ids = sorted(n.split("/")[-1][:-4] for n in zp.namelist() if n.endswith(".pfs"))
    imgs, masks = [], []
    for i in ids:
        im = leer_imagen(images_dir, i)
        imgs.append(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA))
        masks.append(rasterize(parse_pfs(zp.read(f"points/{i}.pfs").decode("latin-1")), size))
    imgs, masks = np.stack(imgs), np.stack(masks)
    pulmon = imgs[(masks[:, 0] | masks[:, 1]) > 0].mean()
    corazon = imgs[masks[:, 2] > 0].mean()
    invertidas = bool(pulmon > corazon)
    if invertidas:
        imgs = 255 - imgs
    return ids, imgs, masks, invertidas
