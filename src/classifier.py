"""Caja negra: ResNet-50 preentrenada en ImageNet que predice cardiomegalia (experimentos E2 y E4).

  python -m src.classifier --data data --run E2_resnet_paap_s0 --seed 0
  python -m src.classifier --data data --run E4_resnet_pa_s0 --views PA --seed 0     # E4: solo placas PA

Decisiones fijas (y por qué):
  - usa las MISMAS imágenes de 256 px y la MISMA partición por paciente que la U-Net: la comparación E2 es justa;
  - la radiografía gris se repite en 3 canales y se normaliza como ImageNet, porque así se preentrenó la red;
  - aumentación igual a la de la U-Net (sin volteo horizontal: el corazón está a la izquierda);
  - BCE con pos_weight = negativos / positivos de train (27 % de cardiomegalia);
  - el mejor epoch se elige con el AUC de la validación NIH; el umbral de decisión, con el índice de Youden en esa
    misma validación. El test oficial no se mira hasta el final;
  - --views PA entrena y valida solo con PA, pero predice TODO el subconjunto: así E4 mide qué pasa en AP sin haberlas
    visto.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .dataset import SegDataset
from .train_seg import leer_subset, set_seed

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ClsDataset(Dataset):
    """Reutiliza la lectura y la aumentación de SegDataset; devuelve (imagen 3 canales, etiqueta)."""

    def __init__(self, images, masks, idx, labels, augment=False, seed=0):
        self.seg = SegDataset(images, masks, idx, augment=augment, seed=seed)
        self.labels = np.asarray(labels, np.float32)

    def __len__(self):
        return len(self.seg)

    def __getitem__(self, i):
        x, _ = self.seg[i]
        x = x * 0.25 + 0.5                       # deshace la normalización de la U-Net → [0, 1]
        x = (x.expand(3, -1, -1) - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.tensor(self.labels[i])


def build_model(arch: str = "resnet50", pretrained: bool = True) -> nn.Module:
    import torchvision
    if arch == "resnet50":
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=w)
    elif arch == "resnet18":   # solo para pruebas rápidas en CPU
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
    else:
        raise ValueError(arch)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def auc(y, p) -> float:
    """AUC por rangos (Mann-Whitney), sin depender de sklearn."""
    y, p = np.asarray(y), np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    orden = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    r = np.empty(len(orden)); r[orden] = np.arange(1, len(orden) + 1)
    todos = np.concatenate([pos, neg])
    for v in np.unique(todos):                  # empates: rango promedio
        k = todos == v
        if k.sum() > 1:
            r[k] = r[k].mean()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def youden(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p, float)
    mejor, umbral = -1.0, 0.5
    for t in np.unique(p):
        pred = p >= t
        j = pred[y == 1].mean() + (~pred[y == 0]).mean() - 1
        if j > mejor:
            mejor, umbral = j, float(t)
    return umbral


def metricas(y, p, umbral) -> dict:
    y, p = np.asarray(y), np.asarray(p, float)
    pred = p >= umbral
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); esp = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    return {"n": len(y), "auc": auc(y, p), "sens": sens, "esp": esp, "f1": 2 * prec * sens / max(prec + sens, 1e-9),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


@torch.no_grad()
def predecir(model, ds, device, bs=64) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            out.append(torch.sigmoid(model(x.to(device)).float()).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--views", default="PA,AP", help="vistas usadas para entrenar y validar (E4: PA)")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18"])
    ap.add_argument("--no-pretrained", action="store_true", help="solo pruebas sin internet")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="solo para pruebas rápidas")
    ap.add_argument("--max-eval", type=int, default=0, help="solo para pruebas rápidas")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(args.out, args.run)
    os.makedirs(out_dir, exist_ok=True)
    D = args.data
    img, msk = os.path.join(D, "nih_images.npy"), os.path.join(D, "nih_masks.npy")
    subset = leer_subset(D)
    views = set(args.views.split(","))
    y_all = [int(e["cardio"]) for e in subset]

    def idx(split):
        return [i for i, e in enumerate(subset) if e["split"] == split and e["view"] in views]

    tr, va = idx("train"), idx("val")
    if args.max_train:
        tr = tr[: args.max_train]
    if args.max_eval:
        va = va[: args.max_eval]
    ds_tr = ClsDataset(img, msk, tr, [y_all[i] for i in tr], augment=True, seed=args.seed)
    ds_va = ClsDataset(img, msk, va, [y_all[i] for i in va])
    pos = sum(y_all[i] for i in tr)
    pos_weight = torch.tensor([(len(tr) - pos) / max(pos, 1)], device=device)

    g = torch.Generator().manual_seed(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, generator=g, drop_last=len(ds_tr) > args.bs,
                       num_workers=args.workers, pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    model = build_model(args.arch, not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(dl_tr)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"[{args.run}] vistas={sorted(views)} train={len(tr)} ({pos} cardio) val={len(va)} device={gpu}")
    hist, best, t_train = [], -1.0, 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logit = model(x).squeeze(1)
            loss = loss_fn(logit.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item(); nb += 1
        if device == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        p_va = predecir(model, ds_va, device)
        a = auc(ds_va.labels, p_va)
        hist.append({"epoch": ep, "train_loss": run_loss / max(nb, 1), "val_auc": a, "t_train_s": t_train})
        print(f"ep {ep:3d}  loss {run_loss / max(nb, 1):.4f}  val AUC {a:.4f}  ({t_train / 60:.1f} min)")
        if a > best:
            best = a
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, os.path.join(out_dir, "best.pt"))

    # ---- Evaluación final con el mejor epoch: umbral de Youden en val, métricas en el test oficial ----
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device)["model"])
    umbral = youden(ds_va.labels, predecir(model, ds_va, device))
    todos = list(range(len(subset)))[: args.max_eval or None]
    p_all = predecir(model, ClsDataset(img, msk, todos, [y_all[i] for i in todos]), device)
    with open(os.path.join(out_dir, "pred_resnet.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "cardio", "view", "prob"])
        for k, i in enumerate(todos):
            e = subset[i]
            w.writerow([e["image"], e["split"], e["cardio"], e["view"], float(p_all[k])])
    test = {}
    for nombre, vs in (("todas", {"PA", "AP"}), ("PA", {"PA"}), ("AP", {"AP"})):
        k = [j for j, i in enumerate(todos) if subset[i]["split"] == "test" and subset[i]["view"] in vs]
        test[nombre] = metricas([y_all[todos[j]] for j in k], p_all[k], umbral)
    resumen = {"run": args.run, "args": vars(args), "gpu": gpu, "train_time_min": t_train / 60,
               "best_epoch": max(hist, key=lambda h: h["val_auc"])["epoch"] if hist else None, "val_auc": best,
               "umbral_youden_val": umbral, "test": test, "history": hist}
    json.dump(resumen, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    print(f"[{args.run}] test: " + "  ".join(f"{n} AUC {m['auc']:.4f}" for n, m in test.items()))


if __name__ == "__main__":
    main()
