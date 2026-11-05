"""Figuras de las diapositivas, leídas de results/. Uso: python slides/figuras.py (desde la raíz del repo)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RES = Path("results")
OUT = Path("slides/figs")
OUT.mkdir(parents=True, exist_ok=True)

MEDIR, CLASIF, GRIS = "#2a78d6", "#eb6834", "#8a8984"  # paleta validada (dataviz, slots 1-2)
TINTA, TINTA2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
})


def media(s):
    """'0.8144 ± 0.0008' -> (0.8144, 0.0008)."""
    a, _, b = str(s).partition("±")
    return float(a), float(b or 0)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(OUT / nombre, bbox_inches="tight")
    plt.close(fig)


# 1. AUC por método (E2)
e2 = pd.read_csv(RES / "kaggle-v2/E2_diagnostico.csv", encoding="utf-8")
nombres = {"ResNet_PA+AP": "ResNet-50 (PA + AP)", "ResNet_soloPA": "ResNet-50 (solo PA)",
           "ICT_unet": "ICT · U-Net (débil)", "LR_ICT_vista": "ICT + vista (reg. log.)",
           "ICT_chexmask": "ICT · CheXmask (sin red)", "ICT_unet_ft": "ICT · U-Net + 80 humanas"}
filas = [(nombres[m], *media(a)) for m, a in zip(e2.metodo, e2.auc_todas)]
filas.sort(key=lambda f: f[1])
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for i, (n, m, d) in enumerate(filas):
    c = CLASIF if "ResNet" in n else (GRIS if "sin red" in n else MEDIR)
    ax.errorbar(m, i, xerr=d, fmt="o", ms=8, color=c, ecolor=c, elinewidth=2, capsize=4)
    ax.text(m + d + 0.003, i, f"{m:.3f}", va="center", color=TINTA, fontsize=9)
ax.set_yticks(range(len(filas)), [f[0] for f in filas])
ax.set_xlim(0.78, 0.925)
ax.set_xlabel("AUC (media ± D. E., 3 semillas)")
ax.grid(axis="y", visible=False)
guardar(fig, "auc_metodos.png")

# 2. AUC PA frente a AP (E4)
e4 = pd.read_csv(RES / "kaggle-v2/E4_por_vista.csv", encoding="utf-8")
sel = ["ResNet_PA+AP", "ResNet_soloPA", "ICT_unet"]
fig, ax = plt.subplots(figsize=(4.8, 2.8))
for i, m in enumerate(sel):
    fila = e4[e4.metodo == m].iloc[0]
    pa, ap = media(fila.auc_PA)[0], media(fila.auc_AP)[0]
    c = CLASIF if "ResNet" in m else MEDIR
    ax.plot([ap, pa], [i, i], color=c, lw=2)
    ax.plot(pa, i, "o", ms=8, color=c, mec="white", mew=2)
    ax.plot(ap, i, "o", ms=8, color="white", mec=c, mew=2.5)
    ax.text(pa + 0.004, i + 0.18, f"PA {pa:.3f}", color=TINTA, fontsize=8.5)
    ax.text(ap - 0.004, i + 0.18, f"AP {ap:.3f}", color=TINTA, fontsize=8.5, ha="right")
    ax.text((pa + ap) / 2, i - 0.32, f"−{pa - ap:.3f}", color=TINTA2, fontsize=8.5, ha="center")
ax.set_yticks(range(len(sel)), [nombres[m] for m in sel])
ax.set_ylim(-0.6, len(sel) - 0.3)
ax.set_xlim(0.74, 0.97)
ax.set_xlabel("AUC por vista (punto lleno: PA · punto vacío: AP)")
ax.grid(axis="y", visible=False)
guardar(fig, "vista_pa_ap.png")

# 3. Fotos con celular: % de decisiones que cambian (E5)
e5 = pd.read_csv(RES / "e5/E5_resumen.csv", encoding="utf-8")
conds = [("iphone16pro_frente", "iPhone 16 Pro\nde frente"), ("iphone16pro_inclinada", "iPhone 16 Pro\ninclinado"),
         ("honor_frente", "Honor X8b\nde frente")]
fig, ax = plt.subplots(figsize=(4.8, 2.9))
ancho = 0.36
for j, (m, etiqueta, c) in enumerate([("ICT_unet", "ICT (U-Net)", MEDIR), ("ResNet_PA+AP", "ResNet-50", CLASIF)]):
    for i, (cond, _) in enumerate(conds):
        v, d = media(e5[(e5.modelo == m) & (e5.condicion == cond)].cambia_decision.iloc[0])
        x = i + (j - 0.5) * (ancho + 0.03)
        ax.bar(x, v * 100, ancho, color=c, label=etiqueta if i == 0 else None)
        ax.text(x, v * 100 + 0.4, f"{v * 100:.1f} %", ha="center", color=TINTA, fontsize=9)
ax.set_xticks(range(len(conds)), [c[1] for c in conds])
ax.set_ylabel("Placas cuya decisión cambia (%)")
ax.set_ylim(0, 16)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="x", visible=False)
guardar(fig, "fotos_cambios.png")

# 4. Grad-CAM: energía en el corazón por grupo (ResNet-50 PA + AP, 3 semillas x 2 vistas)
g = pd.read_csv(RES / "kaggle-v2/E4_gradcam.csv", encoding="utf-8")
g = g[g.run.str.startswith("E2_resnet_paap")]
grupo = {(1, True): "Verdaderos\npositivos", (0, False): "Falsos\npositivos",
         (1, False): "Falsos\nnegativos", (0, True): "Verdaderos\nnegativos"}
g["grupo"] = [grupo[(c, a)] for c, a in zip(g.cardio, g.acierto)]
orden = list(grupo.values())
fig, ax = plt.subplots(figsize=(4.8, 2.9))
for i, gr in enumerate(orden):
    x = g[g.grupo == gr]
    for k, (_, f) in enumerate(x.iterrows()):
        ax.plot(i + (k - 2.5) * 0.06, f.razon_corazon, "o", ms=7, color=CLASIF, mec="white", mew=1.5)
    lo, hi = (round(v + 1e-9, 1) for v in (x.razon_corazon.min(), x.razon_corazon.max()))  # 4.35 -> 4.4
    ax.text(i + 0.25, x.razon_corazon.median(), f"{lo}–{hi}",
            va="center", color=TINTA, fontsize=8.5)
ax.axhline(1, color=GRIS, lw=1, ls="--")
ax.text(3.45, 1.15, "mapa uniforme = 1", color=TINTA2, fontsize=8, ha="right")
ax.set_xticks(range(4), orden)
ax.set_ylim(0, 7)
ax.set_ylabel("Energía de Grad-CAM en el corazón\n(veces la de un mapa uniforme)")
ax.grid(axis="x", visible=False)
guardar(fig, "gradcam_grupos.png")

# 5. Cuatro placas de la tira de Grad-CAM (semilla 0); la 3.ª trae el rótulo PORTABLE
tira = Image.open(RES / "kaggle-v2/E2_resnet_paap_s0_gradcam.png").convert("RGB")
paso = tira.size[0] // 8
Image.fromarray(np.hstack([np.array(tira.crop((k * paso, 0, (k + 1) * paso, tira.size[1])))
                           for k in (0, 2, 4, 5)])).save(OUT / "gradcam_4.png")
print("figuras en", OUT)
