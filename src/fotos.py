"""E5: radiografías fotografiadas desde una pantalla con el celular.

  python -m src.fotos seleccionar --subset data/subset.csv --nih-root <PNG del NIH> --out e5_seleccion   # Kaggle
  python -m src.fotos mostrar --sel e5_seleccion                       # en la laptop, pantalla completa
  python -m src.fotos probar --sel e5_seleccion --foto IMG_0001.jpg    # ¿se leen las marcas de una foto?

Lámina que se muestra en pantalla (lienzo de 1600 × 1000):
  - la radiografía ORIGINAL de 1024 px, reducida a 900 × 900 en el centro;
  - 4 marcas ArUco (DICT_4X4_50, ids 0-3) en las esquinas: con sus 16 esquinas se calcula la homografía que
    devuelve la foto al plano de la pantalla (corrección de perspectiva);
  - 1 marca ArUco (DICT_6X6_250) con el número de placa: las fotos se emparejan solas, sin importar el orden ni el
    nombre del archivo.
Cada foto da dos entradas: "corregida" (homografía) y "sin corregir" (recorte recto del rectángulo que encierra las
marcas, como haría alguien que solo recorta la foto). Ambas se reducen a 256 px igual que en prepare_data.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil

import cv2
import numpy as np

W, H = 1600, 1000                 # lienzo
X0, Y0, LADO = 350, 50, 900       # radiografía
M = 150                           # lado de cada marca
ESQUINAS = {0: (100, 50), 1: (1350, 50), 2: (1350, 800), 3: (100, 800)}   # id -> (x, y) de su esquina sup. izq.
POS_ID = (100, 425)
FONDO, MARGEN = 40, 20            # gris oscuro como un visor; borde blanco para que la marca se detecte
D_ESQ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
D_ID = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
S = 256


def a_256(img: np.ndarray) -> np.ndarray:
    """Misma reducción que prepare_data: gris, INTER_AREA."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- selección (Kaggle)
def seleccionar(subset_csv: str, nih_root: str, out: str, n_por_grupo: int = 25, seed: int = 0):
    """n_por_grupo placas del test oficial por cada combinación vista (PA/AP) × cardiomegalia (0/1)."""
    from .nih import index_images
    filas = [r for r in csv.DictReader(open(subset_csv, newline="")) if r["split"] == "test"]
    rng = random.Random(seed)
    elegidas = []
    for vista in ("PA", "AP"):
        for c in ("0", "1"):
            g = sorted((r for r in filas if r["view"] == vista and r["cardio"] == c), key=lambda r: r["image"])
            elegidas += rng.sample(g, n_por_grupo)
    rng.shuffle(elegidas)                      # orden de la sesión mezclado: el cansancio no cae en un solo grupo
    rutas = index_images(nih_root)
    os.makedirs(os.path.join(out, "png"), exist_ok=True)
    with open(os.path.join(out, "seleccion.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "image", "view", "cardio", "mm_por_px"])
        for k, r in enumerate(elegidas):
            shutil.copy(rutas[r["image"]], os.path.join(out, "png", r["image"]))
            w.writerow([k, r["image"], r["view"], r["cardio"], r["mm_por_px"]])
    print(f"{len(elegidas)} placas en {out}")


def leer_seleccion(sel: str) -> list[dict]:
    return list(csv.DictReader(open(os.path.join(sel, "seleccion.csv"), newline="")))


def original(sel: str, fila: dict) -> np.ndarray:
    return cv2.imread(os.path.join(sel, "png", fila["image"]), cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------- lámina y sesión de fotos
def marca(d, i: int) -> np.ndarray:
    m = cv2.aruco.generateImageMarker(d, i, M)
    return cv2.copyMakeBorder(m, MARGEN, MARGEN, MARGEN, MARGEN, cv2.BORDER_CONSTANT, value=255)


def lamina(img: np.ndarray, k: int, texto: str = "") -> np.ndarray:
    L = np.full((H, W), FONDO, np.uint8)
    L[Y0:Y0 + LADO, X0:X0 + LADO] = cv2.resize(img, (LADO, LADO), interpolation=cv2.INTER_AREA)
    for i, (x, y) in ESQUINAS.items():
        L[y - MARGEN:y + M + MARGEN, x - MARGEN:x + M + MARGEN] = marca(D_ESQ, i)
    x, y = POS_ID
    L[y - MARGEN:y + M + MARGEN, x - MARGEN:x + M + MARGEN] = marca(D_ID, k)
    if texto:
        cv2.putText(L, texto, (1300, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 200, 2, cv2.LINE_AA)
    return L


def mostrar(sel: str, desde: int = 0, x: int = 0, y: int = 0, solo: list[int] | None = None):
    filas = leer_seleccion(sel)
    orden = [n - 1 for n in solo] if solo else list(range(len(filas)))   # solo: números de lámina (1..100) a repetir
    ventana = "E5 - espacio: siguiente | b: anterior | q: salir"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.moveWindow(ventana, x, y)   # con dos monitores: x, y = esquina del monitor elegido (la ventana a pantalla completa no se arrastra)
    cv2.setWindowProperty(ventana, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    p = orden.index(desde) if desde in orden else 0
    while 0 <= p < len(orden):
        k = orden[p]
        cv2.imshow(ventana, lamina(original(sel, filas[k]), k, f"{k + 1}/{len(filas)}"))
        t = cv2.waitKey(0) & 0xFF
        if t in (ord("q"), 27):
            break
        p = max(p - 1, 0) if t == ord("b") else p + 1
    cv2.destroyAllWindows()
    print(f"última placa mostrada: {k}")


# ---------------------------------------------------------------- de la foto a la entrada de 256 px
def leer_foto(ruta: str) -> np.ndarray | None:
    if ruta.lower().endswith((".heic", ".heif")):
        try:
            from pillow_heif import register_heif_opener
            from PIL import Image
            register_heif_opener()
            return cv2.cvtColor(np.array(Image.open(ruta).convert("RGB")), cv2.COLOR_RGB2BGR)
        except ImportError:
            return None
    return cv2.imread(ruta)   # aplica la orientación EXIF


def detectar(gris: np.ndarray, d) -> dict[int, np.ndarray]:
    corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gris)
    return {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def rectificar(foto: np.ndarray, max_lado: int = 2400) -> dict:
    """foto BGR -> {'k', 'n_marcas', 'corregida', 'sin_corregir'} (imágenes de 256 px) o {'motivo': ...}."""
    f = max_lado / max(foto.shape[:2])
    if f < 1:
        foto = cv2.resize(foto, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    gris = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
    esq = {i: c for i, c in detectar(gris, D_ESQ).items() if i in ESQUINAS}
    ids = detectar(gris, D_ID)
    if len(esq) < 2:
        return {"motivo": f"solo {len(esq)} marcas de esquina"}
    if len(ids) != 1:
        return {"motivo": f"{len(ids)} marcas de número"}
    src = np.concatenate([esq[i] for i in sorted(esq)]).astype(np.float32)
    dst = np.concatenate([np.float32([[x, y], [x + M, y], [x + M, y + M], [x, y + M]])
                          for x, y in (ESQUINAS[i] for i in sorted(esq))])
    Hm, _ = cv2.findHomography(src, dst, 0)
    plano = cv2.warpPerspective(gris, Hm, (W, H), flags=cv2.INTER_LINEAR)
    corregida = a_256(plano[Y0:Y0 + LADO, X0:X0 + LADO])
    # sin corregir: rectángulo recto que encierra las marcas detectadas, recortado en la proporción de la lámina
    out = {"k": next(iter(ids)), "n_marcas": len(esq), "corregida": corregida}
    if len(esq) == 4:
        xs, ys = src[:, 0], src[:, 1]
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        ax, ay = (x2 - x1) / (1400), (y2 - y1) / (900)            # marcas: x de 100 a 1500, y de 50 a 950
        r = gris[int(y1 + (Y0 - 50) * ay):int(y1 + (Y0 + LADO - 50) * ay),
                 int(x1 + (X0 - 100) * ax):int(x1 + (X0 + LADO - 100) * ax)]
        out["sin_corregir"] = a_256(r) if r.size else None
    return out


# ---------------------------------------------------------------- degradaciones sintéticas (respaldo de E5)
def sinteticas(img: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Sobre la radiografía a 900 px (como en pantalla) -> entradas de 256 px. Una severidad fija por tipo."""
    rng = np.random.default_rng(seed)
    x = cv2.resize(img, (LADO, LADO), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = {}
    d = rng.uniform(-0.08, 0.08, (4, 2)) * LADO                       # foto inclinada sin corregir
    c = np.float32([[0, 0], [LADO, 0], [LADO, LADO], [0, LADO]])
    out["sint_perspectiva"] = cv2.warpPerspective(x, cv2.getPerspectiveTransform(c, c + d.astype(np.float32)),
                                                  (LADO, LADO), borderValue=FONDO)
    R = cv2.getRotationMatrix2D((LADO / 2, LADO / 2), rng.choice([-1, 1]) * 7, 1.0)
    out["sint_rotacion"] = cv2.warpAffine(x, R, (LADO, LADO), borderValue=FONDO)
    out["sint_desenfoque"] = cv2.GaussianBlur(x, (0, 0), 3)
    out["sint_brillo"] = 255 * ((x / 255) ** 0.6) * 0.8 + 30          # pantalla brillante, contraste lavado
    yy, xx = np.mgrid[:LADO, :LADO]
    out["sint_moire"] = x + 20 * np.sin(2 * np.pi * (xx * 0.95 + yy * 0.31) / 6.5)
    _, b = cv2.imencode(".jpg", np.clip(x, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 15])
    out["sint_jpeg"] = cv2.imdecode(b, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return {k: a_256(np.clip(v, 0, 255).astype(np.uint8)) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("seleccionar")
    a.add_argument("--subset", required=True)
    a.add_argument("--nih-root", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n-por-grupo", type=int, default=25)
    a = sub.add_parser("mostrar")
    a.add_argument("--sel", required=True)
    a.add_argument("--desde", type=int, default=0, help="número de placa para retomar una sesión cortada")
    a.add_argument("--x", type=int, default=0, help="esquina izquierda del monitor donde abrir (p. ej. 1920 = segundo monitor)")
    a.add_argument("--y", type=int, default=0)
    a.add_argument("--solo", type=lambda s: [int(n) for n in s.split(",")], help="solo estas láminas, p. ej. 8,20,21")
    a = sub.add_parser("probar")
    a.add_argument("--sel", required=True)
    a.add_argument("--foto", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "seleccionar":
        seleccionar(args.subset, args.nih_root, args.out, args.n_por_grupo)
    elif args.cmd == "mostrar":
        mostrar(args.sel, args.desde, args.x, args.y, args.solo)
    else:
        foto = leer_foto(args.foto)
        if foto is None:
            raise SystemExit("no se pudo leer la foto (¿HEIC? en el iPhone: Ajustes > Cámara > Formatos > Más compatible)")
        r = rectificar(foto)
        if "motivo" in r:
            raise SystemExit(f"NO se pudo leer: {r['motivo']}")
        orig = a_256(original(args.sel, leer_seleccion(args.sel)[r["k"]]))
        dif = np.abs(r["corregida"].astype(float) - orig).mean()
        print(f"placa {r['k']} · {r['n_marcas']}/4 marcas · diferencia media con el original {dif:.1f} niveles de gris")
        cv2.imwrite(os.path.splitext(args.foto)[0] + "_corregida.png", np.hstack([orig, r["corregida"]]))


if __name__ == "__main__":
    main()
