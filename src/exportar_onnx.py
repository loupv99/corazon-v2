"""Exporta la U-Net a ONNX para la app web (ONNX Runtime Web) y comprueba que da lo mismo que PyTorch.

  python -m src.exportar_onnx --ckpt runs/E1b_debil_rca07_s1/best.pt --out app/modelo/unet.onnx [--png placa.png]

Entrada "imagen": (1, 1, 256, 256) float32 ya normalizada: (gris/255 − 0,5)/0,25, como en el entrenamiento.
Salida "prob": (1, 3, 256, 256) probabilidades (sigmoide incluida) en el orden pulmón der., pulmón izq., corazón.
La semilla se elige por Dice de validación, nunca por el test.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from . import ctr, fotos
from .unet import UNet


class ConSigmoide(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def cargar(ckpt: str) -> torch.nn.Module:
    d = torch.load(ckpt, map_location="cpu")
    a = d.get("args", {})
    net = UNet(in_ch=1, n_out=3, base=a.get("base", 32), depth=a.get("depth", 4))
    net.load_state_dict(d["model"])
    return ConSigmoide(net).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="app/modelo/unet.onnx")
    ap.add_argument("--png", default="", help="radiografía para comparar PyTorch vs. ONNX (si no, ruido)")
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    m = cargar(args.ckpt)
    if args.png:
        g = fotos.a_256(cv2.imread(args.png, cv2.IMREAD_GRAYSCALE))
        x = torch.from_numpy((g.astype(np.float32) / 255 - 0.5) / 0.25)[None, None]
    else:
        x = torch.randn(1, 1, 256, 256)
    torch.onnx.export(m, (x,), args.out, input_names=["imagen"], output_names=["prob"], opset_version=17,
                      dynamo=False)

    import onnxruntime as ort
    p_ort = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"]).run(None, {"imagen": x.numpy()})[0]
    with torch.no_grad():
        p_pt = m(x).numpy()
    dif = float(np.abs(p_ort - p_pt).max())
    mascara_igual = float(((p_ort >= 0.5) == (p_pt >= 0.5)).mean())
    r_pt, r_ort = ctr.medir((p_pt[0] >= 0.5).astype(np.uint8)), ctr.medir((p_ort[0] >= 0.5).astype(np.uint8))
    corrida = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    info = {"corrida": corrida, "onnx": os.path.basename(args.out), "mb": round(os.path.getsize(args.out) / 1e6, 1),
            "max_dif_prob": dif, "pixeles_iguales": mascara_igual, "ICT_pytorch": r_pt["ICT"], "ICT_onnx": r_ort["ICT"]}
    json.dump(info, open(os.path.splitext(args.out)[0] + "_verificacion.json", "w"), indent=1)
    print(info)
    assert dif < 1e-3 and mascara_igual > 0.9999, "ONNX no reproduce a PyTorch"


if __name__ == "__main__":
    main()
