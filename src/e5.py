"""E5: cuánto se mueven el ICT y la ResNet cuando la radiografía llega como foto de celular (corre en CPU).

  python -m src.e5 --sel e5_seleccion --fotos fotos --runs runs --out results/e5

--fotos: una subcarpeta por celular y condición (p. ej. iphone16pro_frente/, iphone16pro_inclinada/, honor_frente/).
Condiciones evaluadas, todas sobre las MISMAS placas:
  original                 la PNG del NIH reducida a 256 px (referencia)
  <carpeta>                foto con corrección de perspectiva (homografía con las marcas ArUco)
  <carpeta>_sin_corregir   la misma foto, solo recortada
  sint_*                   degradaciones sintéticas de src.fotos.sinteticas (respaldo)
Modelos: U-Net E1b (ICT, umbral clínico 0,50) y ResNet-50 PA + AP (umbral de Youden de su validación), 3 semillas.
Métricas por condición frente a la original de la misma semilla: |ΔICT|, % de decisiones que cambian, |Δprob|,
AUC (con 100 placas el intervalo es ancho: se reporta como apoyo, no como resultado principal) y placas no medibles.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from . import ctr, fotos
from .classifier import auc, build_model
from .unet import UNet

FOTO_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif")


def semilla(ckpt: str) -> int:
    return int(re.search(r"_s(\d+)$", os.path.basename(os.path.dirname(ckpt))).group(1))


def entradas(sel: str, dir_fotos: str) -> tuple[dict, pd.DataFrame]:
    """{condicion: {k: imagen 256}} y el registro de cada foto (si se leyó, cuántas marcas, por qué falló)."""
    filas = fotos.leer_seleccion(sel)
    cond = {"original": {}}
    for f in filas:
        k = int(f["k"])
        img = fotos.original(sel, f)
        cond["original"][k] = fotos.a_256(img)
        for nombre, x in fotos.sinteticas(img, seed=k).items():
            cond.setdefault(nombre, {})[k] = x
    log = []
    for d in sorted(glob.glob(os.path.join(dir_fotos, "*"))) if dir_fotos else []:
        if not os.path.isdir(d):
            continue
        c = os.path.basename(d)
        for ruta in sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(FOTO_EXT)):
            if ruta.endswith("_corregida.png"):
                continue
            foto = fotos.leer_foto(ruta)
            r = {"motivo": "no se pudo abrir"} if foto is None else fotos.rectificar(foto)
            log.append({"condicion": c, "archivo": os.path.basename(ruta), "k": r.get("k"),
                        "n_marcas": r.get("n_marcas", 0), "ok": "motivo" not in r, "motivo": r.get("motivo", "")})
            if "motivo" in r or r["k"] >= len(filas):
                continue
            # si una placa se fotografió dos veces, vale la última (la repetición suele ser porque la primera salió mal)
            cond.setdefault(c, {})[r["k"]] = r["corregida"]
            if r.get("sin_corregir") is not None:
                cond.setdefault(f"{c}_sin_corregir", {})[r["k"]] = r["sin_corregir"]
    return cond, pd.DataFrame(log)


@torch.no_grad()
def inferir(cond: dict, filas: list[dict], runs: str, unet_pat: str, resnet_pat: str) -> pd.DataFrame:
    meta = {int(f["k"]): f for f in filas}
    out = []
    for ck in sorted(glob.glob(os.path.join(runs, f"{unet_pat}_s*", "best.pt"))):
        s = semilla(ck)
        d = torch.load(ck, map_location="cpu")
        a = d.get("args", {})
        net = UNet(in_ch=1, n_out=3, base=a.get("base", 32), depth=a.get("depth", 4))
        net.load_state_dict(d["model"]); net.eval()
        for c, imgs in cond.items():
            ks = sorted(imgs)
            x = torch.from_numpy((np.stack([imgs[k] for k in ks]).astype(np.float32) / 255 - 0.5) / 0.25)[:, None]
            m = torch.cat([(torch.sigmoid(net(b)) >= 0.5).to(torch.uint8) for b in x.split(16)]).numpy()
            for j, k in enumerate(ks):
                out.append({"modelo": "ICT_unet", "semilla": s, "condicion": c, "k": k,
                            "score": ctr.medir(m[j], float(meta[k]["mm_por_px"]))["ICT"]})
        print(f"U-Net semilla {s}: listo")
    for ck in sorted(glob.glob(os.path.join(runs, f"{resnet_pat}_s*", "best.pt"))):
        s = semilla(ck)
        net = build_model("resnet50", pretrained=False)
        net.load_state_dict(torch.load(ck, map_location="cpu")["model"]); net.eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        for c, imgs in cond.items():
            ks = sorted(imgs)
            x = torch.from_numpy(np.stack([imgs[k] for k in ks]).astype(np.float32) / 255)[:, None].expand(-1, 3, -1, -1)
            p = torch.cat([torch.sigmoid(net((b - mean) / std)).squeeze(1) for b in x.split(16)]).numpy()
            for j, k in enumerate(ks):
                out.append({"modelo": "ResNet_PA+AP", "semilla": s, "condicion": c, "k": k, "score": float(p[j])})
        print(f"ResNet semilla {s}: listo")
    df = pd.DataFrame(out)
    df["score"] = df.score.replace([np.inf, -np.inf], np.nan)
    df["cardio"] = df.k.map(lambda k: int(meta[k]["cardio"]))
    df["view"] = df.k.map(lambda k: meta[k]["view"])
    return df


def resumir(df: pd.DataFrame, runs: str, resnet_pat: str) -> pd.DataFrame:
    umbral = {("ICT_unet", s): 0.5 for s in df.semilla.unique()}
    for f in glob.glob(os.path.join(runs, f"{resnet_pat}_s*", "results.json")):
        j = json.load(open(f))
        umbral[("ResNet_PA+AP", semilla(f))] = j["umbral_youden_val"]
    ref = df[df.condicion == "original"].set_index(["modelo", "semilla", "k"]).score
    filas = []
    for (mod, s, c), g in df.groupby(["modelo", "semilla", "condicion"]):
        r = ref.loc[mod, s].reindex(g.k).values
        v = g.score.values
        ok = ~np.isnan(v) & ~np.isnan(r)
        u = umbral[(mod, s)]
        filas.append({"modelo": mod, "semilla": s, "condicion": c, "n": len(g), "no_medibles": int(np.isnan(v).sum()),
                      "delta_abs_media": np.abs(v[ok] - r[ok]).mean(), "delta_abs_mediana": np.median(np.abs(v[ok] - r[ok])),
                      "sesgo": (v[ok] - r[ok]).mean(), "cambia_decision": ((v[ok] >= u) != (r[ok] >= u)).mean(),
                      "auc": auc(g.cardio.values[ok], v[ok])})
    return pd.DataFrame(filas)


def pareado(df: pd.DataFrame, runs: str, resnet_pat: str) -> pd.DataFrame:
    """Corregida vs. sin corregir sobre las MISMAS placas: "sin corregir" solo existe cuando se vieron las 4 marcas."""
    partes = []
    for c in sorted(df.condicion.unique()):
        if not c.endswith("_sin_corregir"):
            continue
        base = c[: -len("_sin_corregir")]
        ks = set(df[df.condicion == c].k)
        sub = df[df.condicion.isin(["original", base, c]) & df.k.isin(ks)]
        partes.append(resumir(sub, runs, resnet_pat).assign(fotos=base))
    return pd.concat(partes) if partes else pd.DataFrame()


def tabla(det: pd.DataFrame, por: list[str]) -> pd.DataFrame:
    num = ["n", "no_medibles", "delta_abs_media", "delta_abs_mediana", "sesgo", "cambia_decision", "auc"]
    g = det.groupby(por)[num]
    return g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True, help="carpeta de src.fotos seleccionar (seleccion.csv + png/)")
    ap.add_argument("--fotos", default="", help="carpeta con una subcarpeta por celular y condición")
    ap.add_argument("--runs", required=True, help="carpeta con los best.pt de Kaggle")
    ap.add_argument("--out", default="results/e5")
    ap.add_argument("--unet", default="E1b_debil_rca07")
    ap.add_argument("--resnet", default="E2_resnet_paap")
    ap.add_argument("--solo-tablas", action="store_true", help="rehace las tablas desde E5_por_placa.csv, sin inferir")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    placa = os.path.join(args.out, "E5_por_placa.csv")
    if args.solo_tablas:
        df = pd.read_csv(placa)
    else:
        cond, log = entradas(args.sel, args.fotos)
        if len(log):
            log.to_csv(os.path.join(args.out, "E5_fotos_leidas.csv"), index=False)
            print(log.groupby("condicion").ok.agg(["sum", "size"]).rename(columns={"sum": "leidas", "size": "fotos"}))
            for c, g in log[~log.ok].groupby("condicion"):
                print(f"  {c}: no leídas -> {', '.join(g.archivo + ' (' + g.motivo + ')')}")
        print({c: len(v) for c, v in cond.items()})
        df = inferir(cond, fotos.leer_seleccion(args.sel), args.runs, args.unet, args.resnet)
        df.to_csv(placa, index=False)
    det = resumir(df, args.runs, args.resnet)
    det.to_csv(os.path.join(args.out, "E5_por_semilla.csv"), index=False)
    t = tabla(det, ["modelo", "condicion"])
    t.to_csv(os.path.join(args.out, "E5_resumen.csv"))
    par = pareado(df, args.runs, args.resnet)
    pd.set_option("display.width", 250)
    print(t.to_string())
    if len(par):
        tp = tabla(par, ["fotos", "modelo", "condicion"])
        tp.to_csv(os.path.join(args.out, "E5_pareado_sin_corregir.csv"))
        print("\nCorregida vs. sin corregir, mismas placas (las que tienen las 4 marcas):")
        print(tp.to_string())


if __name__ == "__main__":
    main()
