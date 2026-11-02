# Datos (no se suben al repositorio)

| Fuente | Qué se usa | Dónde | Licencia |
|---|---|---|---|
| NIH ChestX-ray14 | PNG + `Data_Entry_2017.csv` + `test_list.txt` | Kaggle `nih-chest-xrays/data` o https://nihcc.app.box.com/v/ChestXray-NIHCC | "There are no restrictions on the use of the NIH chest x-ray images" + atribución (Google Cloud Healthcare, página del dataset) |
| CheXmask | `OriginalResolution/ChestX-Ray8.csv` (2,0 GB) | https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/ | CC BY 4.0 |
| SCR | `points.zip` (contornos humanos) | https://zenodo.org/record/7056076 | CC BY 4.0 |
| JSRT | 247 radiografías **limpias** | Kaggle `raddar/nodules-in-chest-xrays-jsrt` (copia PNG no oficial) o el JSRT original | POR CONFIRMAR |

**Ojo:** el `jpg.zip` de SCR en Zenodo trae los contornos **dibujados sobre la imagen**. Usarlo como entrada del modelo
filtraría la respuesta. `src/scr.py` solo toma los contornos de `points.zip` y las imágenes de una copia limpia de JSRT.

Verificación de conteos y contenido: `python -I ../scripts/verificar_datos.py` (desde la carpeta del proyecto).
