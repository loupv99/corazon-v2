"""Etiquetas débiles: máscaras automáticas de CheXmask (HybridGNet) para ChestX-ray14.

Formato verificado el 2026-11-02 con 156 filas reales de `OriginalResolution/ChestX-Ray8.csv` y con el
decodificador oficial (github.com/ngaggion/CheXmask-Database, DataPostprocessing/utils.py):
  - columnas: Image Index, Dice RCA (Mean), Dice RCA (Max), Landmarks, Left Lung, Right Lung, Heart, Height, Width;
  - cada máscara es un RLE "inicio longitud inicio longitud ...", con inicio en 1 y píxeles en orden por filas;
  - "Right Lung" es el pulmón derecho del paciente, que en la placa PA aparece a la IZQUIERDA de la imagen.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

# Orden de canales en todo el proyecto: 0 pulmón derecho, 1 pulmón izquierdo, 2 corazón.
COLS = ("Right Lung", "Left Lung", "Heart")
RCA_RECOMENDADO = 0.7   # umbral de calidad que recomiendan los autores de CheXmask


def rle_decode(rle: str, height: int, width: int) -> np.ndarray:
    """RLE de CheXmask -> máscara binaria (H, W) uint8 con valores 0/1."""
    flat = np.zeros(height * width, dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        nums = np.asarray(rle.split(), dtype=np.int64)
        for start, length in zip(nums[0::2] - 1, nums[1::2]):   # -1: el formato empieza en 1
            flat[start:start + length] = 1
    return flat.reshape(height, width)


def row_to_masks(row: dict) -> np.ndarray:
    """Fila del CSV -> máscaras (3, H, W) en el orden de COLS."""
    h, w = int(row["Height"]), int(row["Width"])
    return np.stack([rle_decode(row[c], h, w) for c in COLS])


def iter_rows(path: str, wanted: set[str] | None = None):
    """Recorre el CSV de 2 GB fila por fila (sin cargarlo entero) y devuelve solo las imágenes pedidas."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))   # las celdas RLE superan el límite por defecto
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if wanted is None or row["Image Index"] in wanted:
                yield row
