"""Índice cardiotorácico (ICT) a partir de máscaras de pulmones y corazón.

ICT = (MRD + MLD) / ID  (Chamveha et al., 2020):
  ID   diámetro interno del tórax: la fila con mayor distancia entre el borde externo del pulmón derecho
       (izquierda de la imagen) y el borde externo del pulmón izquierdo (derecha de la imagen);
  línea media: punto medio de ID en esa fila;
  MRD  distancia máxima de la línea media al borde derecho del corazón (izquierda de la imagen);
  MLD  distancia máxima de la línea media al borde izquierdo del corazón (derecha de la imagen).
MRD y MLD se reportan aparte (experimento E6) y, con el espaciado de píxel, en cm.
"""
from __future__ import annotations

import cv2
import numpy as np


def mayor_componente(mask: np.ndarray) -> np.ndarray:
    """Se queda con la región conectada más grande: borra islas sueltas que inflarían los anchos."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def medir(masks: np.ndarray, mm_por_px: float | None = None) -> dict:
    """masks (3, H, W) binarias en el orden pulmón der., pulmón izq., corazón -> medidas del ICT.

    Devuelve NaN si falta alguna estructura (la app debe avisar en vez de medir).
    """
    pd_, pi, cor = (mayor_componente(m) for m in masks)
    nan = {"ICT": np.nan, "MRD": np.nan, "MLD": np.nan, "ID": np.nan, "linea_media": np.nan}
    if pd_.sum() == 0 or pi.sum() == 0 or cor.sum() == 0:
        return nan

    # Tórax: por fila, del borde más a la izquierda del pulmón derecho al más a la derecha del izquierdo.
    filas = np.where(pd_.any(1) & pi.any(1))[0]
    if len(filas) == 0:
        return nan
    izq = np.array([np.where(pd_[r])[0].min() for r in filas])
    der = np.array([np.where(pi[r])[0].max() for r in filas])
    j = int(np.argmax(der - izq))
    id_px = float(der[j] - izq[j])
    medio = (der[j] + izq[j]) / 2.0

    cols = np.where(cor.any(0))[0]
    mrd = max(0.0, medio - cols.min())          # el corazón puede no cruzar la línea media
    mld = max(0.0, cols.max() - medio)
    out = {"ICT": (mrd + mld) / id_px, "MRD": mrd, "MLD": mld, "ID": id_px, "linea_media": medio}
    if mm_por_px:
        out.update({"MRD_cm": mrd * mm_por_px / 10, "MLD_cm": mld * mm_por_px / 10, "ID_cm": id_px * mm_por_px / 10})
    return out


if __name__ == "__main__":
    # Caso sintético con respuesta conocida: tórax de 160 px y corazón de 80 px centrado en la línea media.
    m = np.zeros((3, 256, 256), np.uint8)
    m[0, 60:200, 48:120] = 1      # pulmón derecho (izquierda de la imagen), borde externo en x=48
    m[1, 60:200, 136:208] = 1     # pulmón izquierdo, borde externo en x=207
    m[2, 120:190, 88:168] = 1     # corazón de x=88 a x=167
    r = medir(m, mm_por_px=0.143 * 4)
    print({k: round(v, 3) for k, v in r.items()})
    assert abs(r["ID"] - 159) < 1e-6 and abs(r["linea_media"] - 127.5) < 1e-6
    assert abs(r["MRD"] - 39.5) < 1e-6 and abs(r["MLD"] - 39.5) < 1e-6
    assert abs(r["ICT"] - 79 / 159) < 1e-6
    print("ctr.py OK")
