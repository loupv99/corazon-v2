"""Tablas de E2 (diagnóstico) y E4 (atajo de proyección) a partir de las corridas guardadas en runs/.

  python -m src.analisis --data data --runs runs --out results

Métodos comparados (todos sobre el MISMO test oficial NIH, partición por paciente):
  ICT_unet        ICT medido con la U-Net débil RCA >= 0,7 (E1b), una corrida por semilla
  ICT_unet_ft     ICT con la U-Net ajustada a contornos humanos (E1c)
  ICT_chexmask    ICT medido directamente sobre las máscaras de CheXmask (sin red; línea base geométrica)
  LR_ICT_vista    regresión logística sobre [ICT_unet, AP]: corrige la magnificación AP de forma explícita
  ResNet_PA+AP    caja negra entrenada con ambas vistas
  ResNet_soloPA   caja negra entrenada solo con PA (E4)
Umbrales: el del ICT clínico es 0,50 (fijo, no se ajusta); además, para todos, el de Youden en la validación NIH.
Las semillas se resumen como media ± DE; ICT_chexmask no tiene semillas.

"Fuga de vista" (E4): AUC del puntaje para separar AP de PA SOLO entre placas sin cardiomegalia. Si el puntaje fuera
independiente de la proyección daría 0,5; cuanto más se aleja, más codifica la vista en vez de la enfermedad.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from .classifier import auc, metricas, youden


def cargar(runs: str, patron: str, col: str) -> dict[int, pd.DataFrame]:
    """{semilla: DataFrame(image, score)} de las corridas runs/<patron>_s<semilla>/."""
    out = {}
    for d in sorted(glob.glob(os.path.join(runs, f"{patron}_s*"))):
        m = re.search(r"_s(\d+)$", d)
        f = os.path.join(d, "ict_unet.csv" if col == "ICT" else "pred_resnet.csv")
        if m and os.path.exists(f):
            d = pd.read_csv(f)[["image", col]].rename(columns={col: "score"})
            d["score"] = d.score.replace([np.inf, -np.inf], np.nan)   # placas no medibles: fuera, y se cuentan
            out[int(m.group(1))] = d
    return out


def lr_ict_vista(df: pd.DataFrame) -> pd.Series:
    from sklearn.linear_model import LogisticRegression
    X = np.c_[df.score.fillna(df.score.median()), (df.view == "AP").astype(float)]
    tr = (df.split == "train").values
    lr = LogisticRegression().fit(X[tr], df.cardio[tr])
    return pd.Series(lr.predict_proba(X)[:, 1], index=df.index)


def evaluar(df: pd.DataFrame, umbral_fijo: float | None) -> dict:
    """df con columnas split, cardio, view, score. Devuelve métricas de test (todas, PA, AP) y la fuga de vista."""
    no_medibles = int((df.split == "test").sum() - ((df.split == "test") & df.score.notna()).sum())
    df = df.dropna(subset=["score"])
    va, te = df[df.split == "val"], df[df.split == "test"]
    u_youden = youden(va.cardio, va.score)
    fila = {"umbral_youden_val": u_youden}
    for nombre, umbral in (("youden", u_youden), ("050", umbral_fijo)):
        if umbral is None:
            continue
        for vista, t in (("todas", te), ("PA", te[te.view == "PA"]), ("AP", te[te.view == "AP"])):
            m = metricas(t.cardio, t.score, umbral)
            if nombre == "youden":
                fila[f"auc_{vista}"] = m["auc"]
            for k in ("sens", "esp", "f1"):
                fila[f"{k}_{nombre}_{vista}"] = m[k]
    neg = te[te.cardio == 0]
    fila["fuga_vista_auc"] = auc((neg.view == "AP").astype(int), neg.score)
    fila["n_test"] = len(te)
    fila["no_medibles_test"] = no_medibles
    return fila


def tabla_e1(runs: str) -> pd.DataFrame:
    """E1 con semillas: Dice en el test dorado SCR y error del ICT, media ± DE por variante."""
    filas = []
    for f in sorted(glob.glob(os.path.join(runs, "E1*_s*", "results.json"))):
        j = json.load(open(f))
        var, s = re.match(r"(.+)_s(\d+)$", j["run"]).groups()
        filas.append({"variante": var, "semilla": int(s), **j["scr_test_dorado"]})
    if not filas:
        return pd.DataFrame()
    d = pd.DataFrame(filas)
    num = ["dice_pulmon_der", "dice_pulmon_izq", "dice_corazon", "dice_medio", "ICT_error_abs_medio"]
    g = d.groupby("variante")[num]
    t = g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)
    t.insert(0, "n_semillas", d.groupby("variante").size())
    return t


def icc_a1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(A,1) de McGraw y Wong: dos evaluadores, efectos aleatorios, acuerdo absoluto."""
    M = np.c_[x, y]
    n, k = M.shape
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def tabla_e3(runs: str) -> pd.DataFrame:
    """E3: acuerdo del ICT de la U-Net con el ICT de contornos humanos (SCR, 147 placas), por corrida."""
    filas = []
    for f in sorted(glob.glob(os.path.join(runs, "E1*_s*", "ict_scr.csv"))):
        d = pd.read_csv(f).replace([np.inf, -np.inf], np.nan).dropna()
        dif = d.ICT_unet - d.ICT_humano
        filas.append({"run": os.path.basename(os.path.dirname(f)), "n": len(d), "icc_a1": icc_a1(d.ICT_unet.values,
                      d.ICT_humano.values), "sesgo": dif.mean(), "lim_inf": dif.mean() - 1.96 * dif.std(),
                      "lim_sup": dif.mean() + 1.96 * dif.std(), "error_abs_medio": dif.abs().mean(),
                      "acuerdo_umbral_050": ((d.ICT_unet >= 0.5) == (d.ICT_humano >= 0.5)).mean()})
    return pd.DataFrame(filas).round(4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    for nombre, t in (("E1_segmentacion", tabla_e1(args.runs)), ("E3_acuerdo_ict", tabla_e3(args.runs))):
        if len(t):
            t.to_csv(os.path.join(args.out, f"{nombre}.csv"), index=nombre == "E1_segmentacion")
            print(f"{nombre}:\n{t.to_string()}")

    sub = pd.read_csv(os.path.join(args.data, "subset.csv"))[["image", "split", "cardio", "view", "ICT_chexmask"]]
    metodos: dict[str, dict[int, pd.DataFrame]] = {
        "ICT_unet": cargar(args.runs, "E1b_debil_rca07", "ICT"),
        "ICT_unet_ft": cargar(args.runs, "E1c_rca07_ft_scr", "ICT"),
        "ResNet_PA+AP": cargar(args.runs, "E2_resnet_paap", "prob"),
        "ResNet_soloPA": cargar(args.runs, "E4_resnet_pa", "prob"),
    }
    filas = []
    for metodo, por_semilla in metodos.items():
        for s, sc in por_semilla.items():
            df = sub.merge(sc, on="image", how="left")
            fijo = 0.5 if metodo.startswith("ICT") else None
            filas.append({"metodo": metodo, "semilla": s, **evaluar(df, fijo)})
            if metodo == "ICT_unet":
                df_lr = df.assign(score=lr_ict_vista(df))
                filas.append({"metodo": "LR_ICT_vista", "semilla": s, **evaluar(df_lr, None)})
    filas.append({"metodo": "ICT_chexmask", "semilla": -1,
                  **evaluar(sub.rename(columns={"ICT_chexmask": "score"}), 0.5)})
    det = pd.DataFrame(filas)
    det.to_csv(os.path.join(args.out, "E2_E4_por_semilla.csv"), index=False)

    num = [c for c in det.columns if c not in ("metodo", "semilla")]
    g = det.groupby("metodo", sort=False)[num]
    res = g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)
    res.insert(0, "n_semillas", det.groupby("metodo", sort=False).size())
    cols_e2 = ["n_semillas", "auc_todas", "sens_050_todas", "esp_050_todas", "sens_youden_todas", "esp_youden_todas",
               "f1_youden_todas"]
    cols_e4 = ["n_semillas", "auc_PA", "auc_AP", "esp_050_PA", "esp_050_AP", "esp_youden_PA", "esp_youden_AP",
               "fuga_vista_auc"]
    res[[c for c in cols_e2 if c in res]].to_csv(os.path.join(args.out, "E2_diagnostico.csv"))
    res[[c for c in cols_e4 if c in res]].to_csv(os.path.join(args.out, "E4_por_vista.csv"))
    print("E2:\n", res[[c for c in cols_e2 if c in res]].to_string())
    print("E4:\n", res[[c for c in cols_e4 if c in res]].to_string())

    # Grad-CAM: energía en el corazón frente al área del corazón (referencia de un mapa uniforme)
    gc = []
    for f in sorted(glob.glob(os.path.join(args.runs, "*", "gradcam.csv"))):
        d = pd.read_csv(f)
        d["run"] = os.path.basename(os.path.dirname(f))
        gc.append(d)
    if gc:
        d = pd.concat(gc)
        d["acierto"] = (d.prob >= 0.5) == (d.cardio == 1)
        t = d.groupby(["run", "view", "cardio", "acierto"]).agg(n=("image", "size"), energia_corazon=("energia_corazon", "mean"),
                                                      area_corazon=("area_corazon", "mean"),
                                                      energia_torax=("energia_torax", "mean"),
                                                      area_torax=("area_torax", "mean")).round(3)
        t["razon_corazon"] = (t.energia_corazon / t.area_corazon).round(2)
        t.to_csv(os.path.join(args.out, "E4_gradcam.csv"))
        print("Grad-CAM:\n", t.to_string())
    json.dump({"metodos": {k: sorted(v) for k, v in metodos.items()}},
              open(os.path.join(args.out, "analisis_corridas.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
