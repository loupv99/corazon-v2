"""Detector de calidad de la app: calibra sus umbrales con las placas de E5 y comprueba que señala imágenes con más error.

  python -m src.calidad --sel e5_seleccion --fotos fotos --onnx app/modelo/unet.onnx --out results/calidad

Cada indicador se calcula sobre la imagen de 256 px que entra a la red y sobre su salida, igual que en app/index.html:
  contraste       desviación estándar del gris (0-255)
  nitidez         varianza del laplaciano de 4 vecinos
  exposicion      fracción de píxeles casi negros (<= 5) o casi blancos (>= 250)
  incertidumbre   píxeles con probabilidad entre 0,2 y 0,8 / píxeles con probabilidad >= 0,5 (los tres canales)
  borde           fracción de los píxeles de pulmón o corazón que tocan el marco de la imagen
Umbral de cada indicador = el percentil 1 o 99 (según el sentido) de las 100 placas ORIGINALES del NIH: la imagen se
señala cuando sale del rango que cubre al 99 % de las placas limpias. Luego se mide si las señaladas (fotos y
degradaciones sintéticas) tienen más |ΔICT| frente a su original que las no señaladas.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import pandas as pd

from . import ctr
from .e5 import entradas

OFICIALES = {"honor_frente", "iphone16pro_frente", "iphone16pro_inclinada"}
AVISAN = ["incertidumbre", "borde"]   # los que usa la app para avisar (los demás no predijeron más error)
SENTIDO = {"contraste": "bajo", "nitidez": "bajo", "exposicion": "alto", "incertidumbre": "alto", "borde": "alto"}


def indicadores(img: np.ndarray, prob: np.ndarray) -> dict:
    g = img.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    est = prob >= 0.5
    union = est.any(0)
    marco = np.zeros_like(union); marco[0, :] = marco[-1, :] = marco[:, 0] = marco[:, -1] = True
    return {"contraste": float(g.std()), "nitidez": float(lap.var()),
            "exposicion": float(((img <= 5) | (img >= 250)).mean()),
            "incertidumbre": float(((prob > 0.2) & (prob < 0.8)).sum() / max(1, est.sum())),
            "borde": float((union & marco).sum() / max(1, union.sum()))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--fotos", required=True)
    ap.add_argument("--onnx", default="app/modelo/unet.onnx")
    ap.add_argument("--out", default="results/calidad")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cond, _ = entradas(a.sel, a.fotos)
    # solo las tres tomas oficiales de E5 (la carpeta de fotos también guarda copias crudas: IPHONE_usb, HONOR_usb...)
    cond = {c: v for c, v in cond.items()
            if c == "original" or c.startswith("sint_") or c.replace("_sin_corregir", "") in OFICIALES}
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    filas = []
    for c, imgs in cond.items():
        for k, img in imgs.items():
            x = ((img.astype(np.float32) / 255 - 0.5) / 0.25)[None, None]
            prob = sess.run(["prob"], {"imagen": x})[0][0]
            m = (prob >= 0.5).astype(np.uint8)
            ict = ctr.medir(m)["ICT"]
            filas.append({"condicion": c, "k": k, "ICT": ict, **indicadores(img, prob)})
    df = pd.DataFrame(filas)
    ref = df[df.condicion == "original"].set_index("k").ICT
    df["dICT"] = (df.ICT - df.k.map(ref)).abs()
    orig = df[df.condicion == "original"]
    umbral = {i: float(np.nanpercentile(orig[i], 1 if s == "bajo" else 99)) for i, s in SENTIDO.items()}
    for i, s in SENTIDO.items():
        df[f"mal_{i}"] = df[i] < umbral[i] if s == "bajo" else df[i] > umbral[i]
    df["aviso_app"] = df[[f"mal_{i}" for i in AVISAN]].any(axis=1)
    df["cambia_decision"] = (df.ICT >= 0.5) != (df.k.map(ref) >= 0.5)
    df["grupo"] = np.where(df.condicion == "original", "original",
                           np.where(df.condicion.str.startswith("sint_"), "sintetica",
                                    np.where(df.condicion.str.endswith("_sin_corregir"), "foto_sin_corregir", "foto")))
    df.to_csv(os.path.join(a.out, "calidad_por_imagen.csv"), index=False)
    filas = []
    for col in [f"mal_{i}" for i in SENTIDO] + ["aviso_app"]:
        for g in ["original", "foto", "foto_sin_corregir", "sintetica"]:
            x = df[df.grupo == g]; m = x[col]
            filas.append({"criterio": col.replace("mal_", ""), "grupo": g, "n": len(x), "senaladas": int(m.sum()),
                          "dICT_med_senaladas": x[m].dICT.median(), "dICT_med_resto": x[~m].dICT.median(),
                          "cambia_senaladas": x[m].cambia_decision.mean(), "cambia_resto": x[~m].cambia_decision.mean()})
    res = pd.DataFrame(filas)
    res.to_csv(os.path.join(a.out, "calidad_resumen.csv"), index=False)
    json.dump(umbral, open(os.path.join(a.out, "umbrales.json"), "w"), indent=1)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
