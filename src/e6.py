"""E6: el componente derecho del ICT (MRD) por separado, en fracción del tórax y en cm (corre en CPU, sin modelos).

  python -m src.e6 --runs results/kaggle-v2/runs --out results/e6

Lee runs/E1b_debil_rca07_s*/ict_unet.csv (test oficial) y, por semilla:
  1. AUC de MRD/ID, MLD/ID e ICT por separado (¿cuánto aporta cada lado?);
  2. MRD/ID y MRD en cm en aciertos y errores del ICT al umbral 0,50 (VP, FN, FP, VN);
  3. falsos negativos con MRD/ID alto: FN cuyo MRD/ID supera el percentil 90 de los verdaderos negativos;
  4. MRD en cm por vista (la AP amplía la silueta: los cm no son comparables entre vistas).
La etiqueta del NIH es "Cardiomegaly", no "crecimiento derecho": E6 describe el componente, no diagnostica el ventrículo
derecho.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from .classifier import auc
from .e5 import semilla


def grupo(cardio: int, ict: float) -> str:
    pos = ict >= 0.5
    return {(1, True): "VP", (1, False): "FN", (0, True): "FP", (0, False): "VN"}[(int(cardio), bool(pos))]


def por_semilla(d: pd.DataFrame, s: int) -> tuple[list, pd.DataFrame]:
    d = d[(d.split == "test")].dropna(subset=["ICT", "MRD", "MLD", "ID"]).copy()
    d["MRD_ID"], d["MLD_ID"] = d.MRD / d.ID, d.MLD / d.ID
    d["grupo"] = [grupo(c, i) for c, i in zip(d.cardio, d.ICT)]
    y = d.cardio.values
    filas = [{"semilla": s, "medida": m, "vista": v, "auc": auc(g.cardio.values, g[m].values), "n": len(g)}
             for m in ("ICT", "MRD_ID", "MLD_ID") for v, g in [("todas", d)] + list(d.groupby("view"))]
    p90 = d[d.grupo == "VN"].MRD_ID.quantile(0.9)
    fn = d[d.grupo == "FN"]
    filas.append({"semilla": s, "medida": "FN_con_MRD_ID_>p90_VN", "vista": "todas", "auc": np.nan,
                  "n": len(fn), "frac": (fn.MRD_ID > p90).mean(), "p90_VN": p90})
    assert len(y) == len(d)
    return filas, d.assign(semilla=s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="results/e6")
    ap.add_argument("--unet", default="E1b_debil_rca07")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    filas, todos = [], []
    for f in sorted(glob.glob(os.path.join(args.runs, f"{args.unet}_s*", "ict_unet.csv"))):
        r, d = por_semilla(pd.read_csv(f), semilla(f))
        filas += r
        todos.append(d)
    if not todos:
        raise SystemExit(f"no hay {args.unet}_s*/ict_unet.csv en {args.runs}")
    res = pd.DataFrame(filas)
    d = pd.concat(todos)

    a = res[res.medida != "FN_con_MRD_ID_>p90_VN"].groupby(["medida", "vista"]).auc
    t_auc = (a.mean().round(3).astype(str) + " ± " + a.std().round(3).astype(str)).unstack()
    q = lambda x: f"{x.median():.3f} [{x.quantile(.25):.3f}–{x.quantile(.75):.3f}]"   # mediana [RIC], 3 semillas juntas
    t_grp = d.groupby(["grupo", "view"]).agg(n=("image", "size"), MRD_ID=("MRD_ID", q), MLD_ID=("MLD_ID", q),
                                              MRD_cm=("MRD_cm", q), ICT=("ICT", q))
    t_grp["n"] = (t_grp.n / d.semilla.nunique()).round(1)   # por semilla
    fn = res[res.medida == "FN_con_MRD_ID_>p90_VN"][["semilla", "n", "frac", "p90_VN"]]

    t_auc.to_csv(os.path.join(args.out, "E6_auc_componentes.csv"))
    t_grp.to_csv(os.path.join(args.out, "E6_aciertos_errores.csv"))
    fn.to_csv(os.path.join(args.out, "E6_fn_mrd_alto.csv"), index=False)
    pd.set_option("display.width", 200)
    print("AUC por componente (media ± DE, 3 semillas):\n", t_auc.to_string(), "\n")
    print("Aciertos y errores del ICT al 0,50 (mediana [RIC]; n por semilla):\n", t_grp.to_string(), "\n")
    print("FN con MRD/ID por encima del p90 de los VN:\n", fn.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
