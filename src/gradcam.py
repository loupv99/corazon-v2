"""Grad-CAM de la ResNet y cuánto de su atención cae sobre el corazón (experimento E4, plausibilidad anatómica).

  python -m src.gradcam --data data --ckpt runs/E2_resnet_paap_s0/best.pt --out runs/E2_resnet_paap_s0

Grad-CAM (Selvaraju et al., 2017) con hooks en la última etapa convolucional (layer4): los pesos de cada mapa son la
media espacial del gradiente del logit de cardiomegalia; el mapa es ReLU(Σ peso · activación), reescalado a 256 px.

Métrica por placa: energía en el corazón = Σ(mapa · máscara del corazón) / Σ(mapa). Se usa la máscara de CheXmask
(independiente de nuestra U-Net). Referencia: un mapa uniforme daría exactamente la fracción de área del corazón, así
que se reporta también esa fracción; la red "mira el corazón" solo si la energía la supera claramente.
"""
from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .classifier import ClsDataset, build_model
from .prepare_data import unpack
from .train_seg import leer_subset


class GradCAM:
    def __init__(self, model, capa):
        self.model, self.act, self.grad = model, None, None
        capa.register_forward_hook(self._guardar)

    def _guardar(self, modulo, entrada, salida):
        # hook sobre el tensor (no sobre el módulo): evita el choque de register_full_backward_hook con los ReLU
        # inplace de los bloques de la ResNet
        self.act = salida
        salida.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """x (B, 3, H, W) → mapas (B, H, W) en [0, 1] y probabilidades (B,)."""
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x).squeeze(1)
        logit.sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        mx = cam.flatten(1).max(1).values.clamp_min(1e-8).view(-1, 1, 1)
        return (cam / mx).detach().cpu().numpy(), torch.sigmoid(logit).detach().cpu().numpy()


def superponer(img: np.ndarray, cam: np.ndarray, heart: np.ndarray) -> np.ndarray:
    color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    o = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, color, 0.4, 0)
    cs, _ = cv2.findContours(np.ascontiguousarray(heart), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cs, -1, (255, 255, 255), 1)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-eval", type=int, default=0, help="solo para pruebas rápidas")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = build_model(ck["args"]["arch"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()                                   # BatchNorm en modo inferencia; los gradientes igual se calculan
    cam = GradCAM(model, model.layer4)

    subset = leer_subset(args.data)
    img_p, msk_p = os.path.join(args.data, "nih_images.npy"), os.path.join(args.data, "nih_masks.npy")
    idx = [i for i, e in enumerate(subset) if e["split"] == args.split and e["has_mask"] == "1"][: args.max_eval or None]
    ds = ClsDataset(img_p, msk_p, idx, [int(subset[i]["cardio"]) for i in idx])
    imgs, msks = np.load(img_p, mmap_mode="r"), np.load(msk_p, mmap_mode="r")

    filas, ejemplos = [], {}
    for a in range(0, len(idx), args.bs):
        x = torch.stack([ds[j][0] for j in range(a, min(a + args.bs, len(idx)))]).to(device)
        mapas, probs = cam(x)
        for k, m in enumerate(mapas):
            i = idx[a + k]
            e = subset[i]
            mk = unpack(np.asarray(msks[i]))
            heart, torax = mk[2].astype(bool), (mk[0] | mk[1] | mk[2]).astype(bool)
            tot = m.sum() + 1e-8
            filas.append({"image": e["image"], "cardio": int(e["cardio"]), "view": e["view"], "prob": float(probs[k]),
                          "energia_corazon": float(m[heart].sum() / tot), "area_corazon": float(heart.mean()),
                          "energia_torax": float(m[torax].sum() / tot), "area_torax": float(torax.mean())})
            # 2 ejemplos por (vista, clase) con prob alta para la figura
            clave = (e["view"], int(e["cardio"]))
            if probs[k] >= 0.5 and len(ejemplos.setdefault(clave, [])) < 2:
                ejemplos[clave].append(superponer(np.asarray(imgs[i]), m, mk[2]))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gradcam.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    fila_fig = [im for clave in sorted(ejemplos) for im in ejemplos[clave]]
    if fila_fig:
        cv2.imwrite(os.path.join(args.out, "gradcam_ejemplos.png"), np.concatenate(fila_fig, 1))
    ec = np.array([r["energia_corazon"] for r in filas]); ac = np.array([r["area_corazon"] for r in filas])
    print(f"Grad-CAM {args.split}: n={len(filas)}  energía en corazón {ec.mean():.3f} (área del corazón {ac.mean():.3f})")


if __name__ == "__main__":
    main()
