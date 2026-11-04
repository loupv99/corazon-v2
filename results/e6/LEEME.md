# E6 — el componente derecho del ICT (MRD) por separado

`python -m src.e6 --runs results/kaggle-v2/runs --out results/e6` · U-Net E1b, 3 semillas, test oficial del NIH,
sin modelos ni GPU (lee los `ict_unet.csv` del notebook 02). MRD = distancia de la línea media al borde derecho del
corazón; MLD = al borde izquierdo; ID = diámetro interno del tórax; ICT = (MRD + MLD)/ID.

## 1. ¿Cuánto discrimina cada lado? (`E6_auc_componentes.csv`, AUC media ± DE)

| Medida | Todas | PA | AP |
|---|---|---|---|
| ICT | 0,814 ± 0,001 | 0,864 ± 0,004 | 0,776 ± 0,003 |
| MLD/ID (lado izquierdo) | 0,735 ± 0,003 | 0,804 ± 0,003 | 0,678 ± 0,008 |
| MRD/ID (lado derecho) | 0,697 ± 0,003 | 0,731 ± 0,005 | 0,671 ± 0,002 |

El ICT coincide con E2/E4 (0,8144; 0,8641; 0,7758). Cada lado por separado discrimina menos que la suma; el derecho es
el más débil.

## 2. Aciertos y errores del ICT al 0,50 (`E6_aciertos_errores.csv`, mediana [RIC], 3 semillas juntas)

| Grupo | Vista | n por semilla | MRD/ID | MRD (cm) | ICT |
|---|---|---|---|---|---|
| VP | PA | 390,0 | 0,188 [0,163–0,217] | 5,36 [4,68–6,10] | 0,562 |
| VP | AP | 591,3 | 0,204 [0,171–0,232] | 5,91 [5,06–6,81] | 0,571 |
| FN | PA | 37,0 | 0,154 [0,129–0,165] | 4,59 [3,69–5,29] | 0,482 |
| FN | AP | 50,7 | 0,138 [0,107–0,172] | 4,08 [3,11–5,23] | 0,481 |
| FP | PA | 205,7 | 0,177 [0,155–0,202] | 5,01 [4,22–5,85] | 0,533 |
| FP | AP | 431,7 | 0,180 [0,157–0,208] | 5,33 [4,56–6,07] | 0,537 |
| VN | PA | 386,3 | 0,137 [0,117–0,158] | 4,18 [3,46–4,85] | 0,451 |
| VN | AP | 261,3 | 0,148 [0,123–0,174] | 4,59 [3,65–5,31] | 0,472 |

## 3. ¿Esconden los falsos negativos un lado derecho grande? (`E6_fn_mrd_alto.csv`)

Fracción de FN cuyo MRD/ID supera el percentil 90 de los VN (≈ 0,187): 15,6 %, 15,7 % y 13,1 % (semillas 0–2; 84–90 FN
por semilla). Si MRD/ID no tuviera relación con la etiqueta se esperaría ~10 %.

## Lectura

1. **El lado derecho no rescata los casos perdidos.** El MRD/ID de los FN (mediana 0,138–0,154) se parece al de los
   VN (0,137–0,148) y queda lejos del de los VP (0,188–0,204). Solo 13–16 % de los FN tienen un lado derecho alto.
2. **Los cm dependen de la vista.** En los cuatro grupos el MRD en cm es mayor en AP que en PA (p. ej., VN 4,59 frente a
   4,18 cm): la proyección AP amplía la silueta. Un umbral en cm solo es válido dentro de una misma vista.
3. **Los FP del ICT son, sobre todo, AP** (432 frente a 206 por semilla), coherente con el sesgo de proyección de E4.
4. Límite: la etiqueta del NIH es "Cardiomegaly", no "crecimiento de cavidades derechas". E6 muestra que la medida
   existe y cómo se distribuye; que sirva para el corazón derecho de altura queda **POR CONFIRMAR** con datos
   etiquetados para eso (ecocardiograma), que este trabajo no tiene.
