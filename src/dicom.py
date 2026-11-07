"""Placas PNG del NIH -> DICOM de captura secundaria (Secondary Capture) para probar la app.

No es el estudio original: el NIH publicó PNG de 8 bits a 1 024 px, no los DICOM. El archivo lleva la misma imagen,
la proyección (PA/AP) y el tamaño del píxel del NIH, con el identificador anonimizado.

Tamaño del píxel: el mismo que usa el proyecto (src/prepare_data.py): mm por píxel a 256 px =
espaciado original x ancho original / 256. Para el PNG de 1 024 px, la columna mide eso x 256 / 1 024. Solo se usa el
espaciado horizontal (MRD, MLD e ID son anchos); como el NIH estiró cada placa a un cuadrado, el vertical puede ser
otro y no se conoce aquí, así que PixelSpacing repite el horizontal (se anota en ImageComments).

Uso:
  python -m src.dicom                      convierte las dos placas de ejemplo de la app y comprueba la conversión
  python -m src.dicom --sel e5_seleccion --imagenes 00028509_016.png 00028445_000.png
                                           convierte placas de la selección de E5 (seleccion.csv trae vista y mm/px)
                                           a app/ejemplos/demo_<vista>_<con|sin>_cardiomegalia.dcm
"""
import argparse
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

RAIZ = Path(__file__).resolve().parent.parent
TABLA = RAIZ / "results/kaggle-v2/runs/E1b_debil_rca07_s1/ict_unet.csv"   # modelo de la app; trae MRD en px y en cm
EJEMPLOS = {"con_cardiomegalia": "00027797_000.png", "sin_cardiomegalia": "00018721_017.png"}


def png_a_dicom(png: Path, salida: Path, mm_col: float, vista: str, imagen_nih: str) -> None:
    g = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(salida), {}, file_meta=meta, preamble=b"\0" * 128)
    hoy = dt.date.today().strftime("%Y%m%d")
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.SOPClassUID, ds.SOPInstanceUID = SecondaryCaptureImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.Modality, ds.ConversionType = "CR", "WSD"        # radiografía digitalizada (captura secundaria)
    ds.PatientName = "Anonimo^NIH"
    ds.PatientID = "NIH-" + imagen_nih.split("_")[0]
    ds.StudyDate = ds.ContentDate = hoy
    ds.StudyDescription = "Torax frontal - ChestX-ray14 (NIH), convertido a DICOM"
    ds.SeriesDescription = f"{vista} - {imagen_nih}"
    ds.BodyPartExamined, ds.ViewPosition = "CHEST", vista
    ds.ImageComments = (f"Convertido de {imagen_nih} (PNG 8 bits del NIH); no es el estudio original. "
                        "PixelSpacing: solo el horizontal es del NIH; el vertical lo repite.")
    ds.SeriesNumber, ds.InstanceNumber = 1, 1
    ds.Rows, ds.Columns = g.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelSpacing = [f"{mm_col:.6f}", f"{mm_col:.6f}"]
    ds.WindowCenter, ds.WindowWidth = 128, 256
    ds.PixelData = g.tobytes()
    ds.save_as(str(salida), enforce_file_format=True)


def comprobar(salida: Path, png: Path, mm256_esperado: float, fila) -> None:
    """Los píxeles vuelven idénticos al PNG y el espaciado reproduce MRD/MLD/ID en cm de la tabla del modelo."""
    import pydicom
    ds = pydicom.dcmread(str(salida))
    igual = np.array_equal(ds.pixel_array, cv2.imread(str(png), cv2.IMREAD_GRAYSCALE))
    mm = float(ds.PixelSpacing[1]) * ds.Columns / 256
    print(f"{salida.name}: {ds.SeriesDescription} | píxeles iguales al PNG: {igual} | "
          f"{salida.stat().st_size // 1024} KB | mm/px a 256: {mm:.4f} (esperado {mm256_esperado:.4f})")
    if fila is not None:
        print(f"   ICT del modelo {fila.ICT:.2f} | MRD {fila.MRD * mm / 10:.2f} cm (tabla {fila.MRD_cm:.2f}), "
              f"MLD {fila.MLD * mm / 10:.2f} cm (tabla {fila.MLD_cm:.2f}), ID {fila.ID * mm / 10:.2f} cm "
              f"(tabla {fila.ID_cm:.2f})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAIZ / "app/ejemplos"))
    ap.add_argument("--sel", help="carpeta de la selección de E5 (png/ + seleccion.csv)")
    ap.add_argument("--imagenes", nargs="+", help="nombres de las placas de --sel a convertir")
    a = ap.parse_args(argv)
    t = pd.read_csv(TABLA).set_index("image")
    if a.sel:
        sel = pd.read_csv(Path(a.sel) / "seleccion.csv").set_index("image")
        for img in a.imagenes:
            r = sel.loc[img]
            png = Path(a.sel) / "png" / img
            ancho = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE).shape[1]
            salida = Path(a.out) / f"demo_{r.view}_{'con' if r.cardio else 'sin'}_cardiomegalia.dcm"
            png_a_dicom(png, salida, r.mm_por_px * 256 / ancho, r.view, img)
            comprobar(salida, png, r.mm_por_px, t.loc[img] if img in t.index else None)
        return
    for nombre, img in EJEMPLOS.items():
        fila = t.loc[img]
        png = RAIZ / f"app/ejemplos/{nombre}.png"
        ancho = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE).shape[1]
        mm256 = fila.MRD_cm * 10 / fila.MRD                       # mm por píxel a 256 px (prepare_data.py)
        salida = Path(a.out) / f"{nombre}.dcm"
        png_a_dicom(png, salida, mm256 * 256 / ancho, fila.view, img)
        comprobar(salida, png, mm256, fila)


if __name__ == "__main__":
    main()
