"""Entrenamiento de la U-Net (experimento E1: etiquetas débiles vs. doradas).

Variantes de E1 (ver run_all.sh):
  python -m src.train_seg --data data --run E1a_debil_todas  --masks all
  python -m src.train_seg --data data --run E1b_debil_rca07  --masks rca07
  python -m src.train_seg --data data --run E1c_rca07_ft_scr --train-source scr --init runs/E1b_debil_rca07/best.pt \
                          --epochs 30 --lr 2e-4

Decisiones fijas (y por qué):
  - el mejor epoch se elige con el Dice medio en la validación NIH (máscaras débiles RCA >= 0,7); en el ajuste con SCR,
    con 20 de las 100 placas doradas de ajuste. El test dorado de SCR (147 placas) y el test NIH no se miran hasta el
    final (lista de verificación del enunciado);
  - checkpoint en cada epoch (last.pt) y reanudación con --resume: en Kaggle una sesión puede cortarse;
  - al terminar se predicen las máscaras de TODO el subconjunto NIH y se calcula el ICT de cada placa (ict_unet.csv),
    que es la entrada del experimento E2 del día siguiente.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import ctr
from .dataset import SegDataset
from .losses import combo_loss
from .metrics import DiceAccumulator
from .prepare_data import unpack
from .unet import UNet, count_params


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def leer_subset(data: str) -> list[dict]:
    with open(os.path.join(data, "subset.csv"), newline="") as f:
        return list(csv.DictReader(f))


def indices(subset, split: str, masks: str) -> list[int]:
    out = []
    for i, e in enumerate(subset):
        if e["split"] != split or e["has_mask"] != "1":
            continue
        if masks == "rca07" and float(e["rca_mean"]) < 0.7:
            continue
        out.append(i)
    return out


@torch.no_grad()
def predecir(model, loader, device):
    """Devuelve las máscaras predichas (N, 3, H, W) uint8."""
    model.eval()
    preds = []
    for x, _ in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            p = torch.sigmoid(model(x.to(device)).float())
        preds.append((p >= 0.5).to(torch.uint8).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros((0, 3, 1, 1), np.uint8)


def evaluar(model, ds, device, bs=32):
    pred = predecir(model, DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0), device)
    acc = DiceAccumulator()
    for k, p in enumerate(pred):
        acc.update(p, ds[k][1].numpy().astype(np.uint8))
    return acc.summary(), pred


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="carpeta de salida de prepare_data")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--masks", default="rca07", choices=["all", "rca07"], help="filtro de calidad de las máscaras débiles")
    ap.add_argument("--train-source", default="nih", choices=["nih", "scr"], help="scr = ajuste con 100 placas doradas")
    ap.add_argument("--init", default="", help="checkpoint inicial (para E1c)")
    ap.add_argument("--resume", default="", help="last.pt de una corrida cortada")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="solo para pruebas rápidas")
    ap.add_argument("--max-eval", type=int, default=0, help="solo para pruebas rápidas")
    ap.add_argument("--stop-after", type=int, default=0, help="solo para probar --resume: corta tras N epochs")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(args.out, args.run)
    os.makedirs(out_dir, exist_ok=True)
    D = args.data
    nih_img, nih_msk = os.path.join(D, "nih_images.npy"), os.path.join(D, "nih_masks.npy")
    if os.path.exists(os.path.join(D, "scr.csv")):
        scr_img, scr_msk = np.load(os.path.join(D, "scr_images.npy")), np.load(os.path.join(D, "scr_masks.npy"))
        with open(os.path.join(D, "scr.csv"), newline="") as f:
            scr_rows = list(csv.DictReader(f))
    else:   # prepare_data se corrió sin imágenes JSRT limpias: no hay test dorado
        print("AVISO: sin SCR; no habrá evaluación con contornos humanos")
        scr_img = scr_msk = np.zeros((0, 1, 1), np.uint8)
        scr_rows = []
        if args.train_source == "scr":
            raise SystemExit("--train-source scr necesita los datos de SCR")
    subset = leer_subset(D)

    if args.train_source == "nih":
        ds_tr = SegDataset(nih_img, nih_msk, indices(subset, "train", args.masks), augment=True, seed=args.seed)
        ds_va = SegDataset(nih_img, nih_msk, indices(subset, "val", "rca07")[: args.max_eval or None])
    else:
        # Ajuste con contornos humanos: el mejor epoch se elige con 20 de las 100 placas doradas, NO con la validación
        # débil de NIH. En la Version #1 de Kaggle se eligió con la débil y ganó siempre el epoch 1: al acercarse al
        # estilo humano el modelo se aleja del de CheXmask, así que esa validación castiga justo lo que se busca.
        ft = [i for i, r in enumerate(scr_rows) if r["split"] == "finetune"]
        random.Random(args.seed).shuffle(ft)
        ds_tr = SegDataset(scr_img, scr_msk, ft[20:], augment=True, seed=args.seed)
        ds_va = SegDataset(scr_img, scr_msk, ft[:20])
    if args.max_train:
        ds_tr.idx = ds_tr.idx[: args.max_train]
    ds_scr = SegDataset(scr_img, scr_msk, [i for i, r in enumerate(scr_rows) if r["split"] == "test"][: args.max_eval or None])

    g = torch.Generator().manual_seed(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, generator=g, drop_last=len(ds_tr) > args.bs,
                       num_workers=args.workers, pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    model = UNet(in_ch=1, n_out=3, base=args.base, depth=args.depth).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(dl_tr)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    ep0, hist, best, t_train = 1, [], -1.0, 0.0
    if args.epochs == 0 and not os.path.exists(os.path.join(out_dir, "best.pt")):
        # --epochs 0 = solo evaluar un best.pt ya entrenado (p. ej. el de la Version #1 de Kaggle) copiado en out_dir
        raise SystemExit(f"--epochs 0 necesita {out_dir}/best.pt")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        if ck["epochs"] != args.epochs:
            # OneCycleLR reparte la tasa de aprendizaje sobre el total de epochs: cambiarlo a mitad de camino rompe
            # el programa. Reanudar = continuar la MISMA corrida.
            raise SystemExit(f"--resume necesita el mismo --epochs de la corrida original ({ck['epochs']})")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        ep0, hist, best, t_train = ck["epoch"] + 1, ck["hist"], ck["best"], ck["t_train"]
        print(f"reanudando desde el epoch {ck['epoch']}")

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"[{args.run}] train={len(ds_tr)} val={len(ds_va)} scr_test={len(ds_scr)} params={count_params(model):,} "
          f"device={gpu}")
    for ep in range(ep0, args.epochs + 1):
        model.train()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
            loss = combo_loss(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item(); nb += 1
        if device == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        val, _ = evaluar(model, ds_va, device)
        hist.append({"epoch": ep, "train_loss": run_loss / max(nb, 1), **{f"val_{k}": v for k, v in val.items()},
                     "t_train_s": t_train})
        print(f"ep {ep:3d}  loss {run_loss / max(nb, 1):.4f}  val Dice corazón {val['dice_corazon']:.4f}  "
              f"medio {val['dice_medio']:.4f}  ({t_train / 60:.1f} min)")
        if val["dice_medio"] > best:
            best = val["dice_medio"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": val},
                       os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "epochs": args.epochs, "hist": hist, "best": best,
                    "t_train": t_train}, os.path.join(out_dir, "last.pt"))
        if args.stop_after and ep >= args.stop_after:
            print(f"corte simulado tras el epoch {ep}")
            return

    # ---- Evaluación final con el mejor epoch: test dorado SCR (E1) y acuerdo del ICT (E3 preliminar) ----
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device)["model"])
    res_scr, pred_scr = evaluar(model, ds_scr, device)
    err = []
    with open(os.path.join(out_dir, "ict_scr.csv"), "w", newline="") as f:   # por placa: entrada de E3
        w = csv.writer(f)
        w.writerow(["id", "ICT_humano", "ICT_unet"])
        for k, i in enumerate(ds_scr.idx):
            ict_pred = ctr.medir(pred_scr[k])["ICT"]
            ict_hum = float(scr_rows[i]["ICT_humano"])
            err.append(abs(ict_pred - ict_hum))
            w.writerow([scr_rows[i]["id"], ict_hum, ict_pred])
    res_scr["ICT_error_abs_medio"] = float(np.nanmean(err)) if err else float("nan")
    res_scr["ICT_no_medible"] = int(np.isnan(err).sum()) if err else 0

    # ---- ICT predicho para todo el subconjunto NIH (entrada de E2) ----
    todos = list(range(len(subset)))[: args.max_eval or None]
    pred_nih = predecir(model, DataLoader(SegDataset(nih_img, nih_msk, todos), batch_size=32, num_workers=0), device)
    with open(os.path.join(out_dir, "ict_unet.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "cardio", "view", "ICT", "MRD", "MLD", "ID", "MRD_cm", "MLD_cm", "ID_cm"])
        for k, i in enumerate(todos):
            e = subset[i]
            m = ctr.medir(pred_nih[k], float(e["mm_por_px"]))
            w.writerow([e["image"], e["split"], e["cardio"], e["view"]] +
                       [m.get(c, np.nan) for c in ("ICT", "MRD", "MLD", "ID", "MRD_cm", "MLD_cm", "ID_cm")])

    resumen = {"run": args.run, "args": vars(args), "gpu": gpu, "train_time_min": t_train / 60,
               "best_val": max(hist, key=lambda h: h["val_dice_medio"]) if hist else None,
               "scr_test_dorado": res_scr, "history": hist}
    json.dump(resumen, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    print(f"[{args.run}] SCR test (dorado): {json.dumps(res_scr)}")


if __name__ == "__main__":
    main()
