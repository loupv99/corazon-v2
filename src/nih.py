"""NIH ChestX-ray14: subconjunto de cardiomegalia y partición por paciente.

Decisiones (y por qué):
  - Positivos: TODAS las imágenes con "Cardiomegaly" en Finding Labels (2 776 en el CSV v2020).
  - Negativos: muestra de imágenes SIN cardiomegalia, con o sin otros hallazgos. Si solo se usaran "No Finding", el
    clasificador podría aprender "enfermo vs. sano" en vez de "corazón grande".
  - Partición por Patient ID: si existe la lista oficial test_list.txt del NIH se usa como test (es por paciente);
    train/val se separan por paciente dentro del resto. Así ningún paciente cae en dos particiones (sin fuga, R4).
  - Se conserva View Position para el experimento del atajo AP (E4): 43,7 % AP en cardiomegalia vs. 34,9 % en
    No Finding, medido con el CSV v2020.
"""
from __future__ import annotations

import csv
import os
import random

CARDIO = "Cardiomegaly"


def load_entries(csv_path: str) -> list[dict]:
    """Lee Data_Entry_2017(_v2020).csv con nombres de columna normalizados."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    cab, out = rows[0], []
    for r in rows[1:]:
        d = dict(zip(cab, r))
        out.append({
            "image": d["Image Index"],
            "patient": d["Patient ID"],
            "labels": d["Finding Labels"],
            "cardio": int(CARDIO in d["Finding Labels"].split("|")),
            "view": d["View Position"],
            "age": d["Patient Age"],
            "sex": d["Patient Gender"],
            # el CSV trae 11 columnas: el ancho/alto original y el espaciado quedan partidos en dos cabeceras
            "orig_w": int(r[7]),
            "spacing_mm": float(r[9]),
        })
    return out


def read_list(path: str | None) -> set[str] | None:
    if not path or not os.path.exists(path):
        return None
    return {l.strip() for l in open(path) if l.strip()}


def build_subset(entries: list[dict], n_neg: int, seed: int = 0, test_list: set[str] | None = None,
                 val_frac: float = 0.15, test_frac: float = 0.15) -> list[dict]:
    """Elige positivos + negativos y asigna split por paciente. Devuelve las filas elegidas con 'split'."""
    rng = random.Random(seed)
    pacientes = sorted({e["patient"] for e in entries})
    pac_cardio = {e["patient"] for e in entries if e["cardio"]}

    if test_list is not None:
        test_p = {e["patient"] for e in entries if e["image"] in test_list}
    else:
        test_p = set()
    resto = [p for p in pacientes if p not in test_p]
    # estratificar por "el paciente tiene alguna cardiomegalia" para que val/test no se queden sin positivos
    split = {p: "test" for p in test_p}
    for grupo in ([p for p in resto if p in pac_cardio], [p for p in resto if p not in pac_cardio]):
        rng.shuffle(grupo)
        n_val = round(len(grupo) * val_frac)
        n_test = 0 if test_list is not None else round(len(grupo) * test_frac)
        for i, p in enumerate(grupo):
            split[p] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    pos = [e for e in entries if e["cardio"]]
    neg = [e for e in entries if not e["cardio"]]
    rng.shuffle(neg)
    elegidas = pos + neg[:n_neg]
    for e in elegidas:
        e["split"] = split[e["patient"]]
    return sorted(elegidas, key=lambda e: e["image"])


def index_images(root: str) -> dict[str, str]:
    """Nombre de archivo -> ruta, buscando en todas las subcarpetas (en Kaggle: images_001/images/...)."""
    idx = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".png"):
                idx[f] = os.path.join(d, f)
    return idx
