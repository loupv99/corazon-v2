"""Prepara todos los datos del proyecto en arrays listos para entrenar (corre una sola vez, ~20 min en Kaggle).

Entrada: CSV de NIH + imágenes, CSV de CheXmask (2 GB) y los zip de SCR.
Salida en --out:
  subset.csv           una fila por imagen elegida: split, cardio, vista, espaciado, RCA, ICT de CheXmask...
  nih_images.npy       (N, S, S) uint8, radiografías reducidas a S px
  nih_masks.npy        (N, S, S) uint8, máscaras débiles empaquetadas en bits: der=1, izq=2, corazón=4
  scr_images.npy / scr_masks.npy / scr.csv   etiquetas doradas (mismo empaquetado)

Ejemplo (Kaggle):
  python -m src.prepare_data --nih-csv /kaggle/input/data/Data_Entry_2017.csv --nih-root /kaggle/input/data \
      --test-list /kaggle/input/data/test_list.txt --chexmask /tmp/raw/ChestX-Ray8.csv \
      --scr-points /tmp/raw/points.zip --scr-images <carpeta con las PNG limpias de JSRT> --out /kaggle/working/data
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import cv2
import numpy as np

from . import chexmask, ctr, nih, scr


def pack(masks: np.ndarray) -> np.ndarray:
    """(3, H, W) binarias -> (H, W) uint8 con un bit por estructura (el corazón se superpone a los pulmones)."""
    return (masks[0] | (masks[1] << 1) | (masks[2] << 2)).astype(np.uint8)


def unpack(packed: np.ndarray) -> np.ndarray:
    """(..., H, W) uint8 -> (..., 3, H, W) uint8."""
    return np.stack([(packed >> k) & 1 for k in range(3)], axis=-3).astype(np.uint8)


def reducir_mascaras(masks: np.ndarray, size: int) -> np.ndarray:
    """1024 -> size con promedio de área y umbral 0,5: conserva mejor los bordes que el vecino más cercano."""
    return np.stack([(cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) >= 0.5)
                     for m in masks]).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-csv", required=True)
    ap.add_argument("--nih-root", required=True, help="carpeta con las PNG (se busca en subcarpetas)")
    ap.add_argument("--test-list", default="", help="test_list.txt oficial del NIH (partición por paciente)")
    ap.add_argument("--chexmask", required=True, help="OriginalResolution/ChestX-Ray8.csv")
    ap.add_argument("--scr-points", required=True, help="points.zip de SCR (contornos humanos)")
    ap.add_argument("--scr-images", default="", help="carpeta con las radiografías JSRT LIMPIAS (no el jpg.zip de "
                                                     "Zenodo, que trae los contornos dibujados). Vacío = sin SCR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-neg", type=int, default=6000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0, help="solo para pruebas rápidas")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # 1. Subconjunto y partición por paciente
    entries = nih.load_entries(args.nih_csv)
    subset = nih.build_subset(entries, args.n_neg, args.seed, nih.read_list(args.test_list))
    rutas = nih.index_images(args.nih_root)
    subset = [e for e in subset if e["image"] in rutas]
    if args.max_images:
        subset = subset[: args.max_images]
    pos = {e["image"]: i for i, e in enumerate(subset)}
    print(f"subconjunto: {len(subset)} imágenes con PNG disponible ({sum(e['cardio'] for e in subset)} cardiomegalia)")

    # 2. Imágenes reducidas
    S, N = args.size, len(subset)
    imgs = np.lib.format.open_memmap(os.path.join(args.out, "nih_images.npy"), "w+", np.uint8, (N, S, S))
    for i, e in enumerate(subset):
        im = cv2.imread(rutas[e["image"]], cv2.IMREAD_GRAYSCALE)   # algunas PNG del NIH traen 4 canales
        imgs[i] = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
        e["mm_por_px"] = e["spacing_mm"] * e["orig_w"] / S        # espaciado de la imagen ORIGINAL -> S px
        if i % 1000 == 0:
            print(f"  imágenes {i}/{N}  {time.time() - t0:.0f}s")
    imgs.flush()

    # 3. Máscaras débiles de CheXmask (lectura en streaming del CSV de 2 GB)
    masks = np.lib.format.open_memmap(os.path.join(args.out, "nih_masks.npy"), "w+", np.uint8, (N, S, S))
    for e in subset:
        e.update({"has_mask": 0, "rca_mean": np.nan, "ICT_chexmask": np.nan})
    n = 0
    for row in chexmask.iter_rows(args.chexmask, set(pos)):
        e = subset[pos[row["Image Index"]]]
        m = reducir_mascaras(chexmask.row_to_masks(row), S)
        masks[pos[row["Image Index"]]] = pack(m)
        medida = ctr.medir(m, e["mm_por_px"])
        e.update({"has_mask": 1, "rca_mean": float(row["Dice RCA (Mean)"]), "ICT_chexmask": medida["ICT"]})
        n += 1
        if n % 1000 == 0:
            print(f"  máscaras {n}/{N}  {time.time() - t0:.0f}s")
    masks.flush()
    print(f"CheXmask: {n}/{N} imágenes con máscara")

    campos = ["image", "patient", "split", "cardio", "labels", "view", "age", "sex", "orig_w", "spacing_mm",
              "mm_por_px", "has_mask", "rca_mean", "ICT_chexmask"]
    with open(os.path.join(args.out, "subset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(subset)

    resumen = {sp: sum(e["split"] == sp for e in subset) for sp in ("train", "val", "test")}
    if not args.scr_images:
        print(f"splits NIH: {resumen} · SCR OMITIDO (falta --scr-images) · listo en {time.time() - t0:.0f}s")
        return

    # 4. Etiquetas doradas (SCR); split fijo: 100 para un posible ajuste (E1c) y 147 de test dorado
    ids, s_img, s_msk, invertidas = scr.load_scr(args.scr_points, args.scr_images, S)
    print(f"SCR: {len(ids)} radiografías{' (escala de grises invertida)' if invertidas else ''}")
    np.save(os.path.join(args.out, "scr_images.npy"), s_img)
    np.save(os.path.join(args.out, "scr_masks.npy"), np.stack([pack(m) for m in s_msk]))
    orden = np.random.RandomState(args.seed).permutation(len(ids))
    split_scr = {ids[k]: ("finetune" if r < 100 else "test") for r, k in enumerate(orden)}
    with open(os.path.join(args.out, "scr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "ICT_humano"])
        for i, sid in enumerate(ids):
            w.writerow([sid, split_scr[sid], ctr.medir(s_msk[i])["ICT"]])
    print(f"splits NIH: {resumen} · SCR: {len(ids)} · listo en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
