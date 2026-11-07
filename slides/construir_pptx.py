"""Diapositivas en PowerPoint EDITABLE sobre la plantilla UNI (slides/plantilla_uni.pptx).

Mismo contenido que slides.tex (23 diapositivas): textos, tablas y diagramas como cuadros y formas de PowerPoint; los
gráficos y fotos como imágenes. Las notas del orador salen de la columna «Idea que debe quedar» de GUION.md.

Uso (desde la raíz del repo): python slides/construir_pptx.py  ->  slides/slides.pptx
Marcado en los textos: **negrita**, [[m|medir]] azul, [[c|clasificar]] naranja, [[g|granate]], [[s|gris]], //cursiva//.
"""
import copy
import re
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

RAIZ = Path(__file__).resolve().parent.parent
SL = RAIZ / "slides"
FIG = SL / "figs"

GR, AZ, TINTA, TINTA2 = (RGBColor.from_string(h) for h in ("7A1518", "1F3B66", "222222", "5A5A5A"))
MED, CLA, LINEA, FONDO, BLANCO = (RGBColor.from_string(h) for h in ("2A78D6", "EB6834", "D8DEE4", "F4F6F8", "FFFFFF"))
COL = {"m": MED, "c": CLA, "g": GR, "s": TINTA2}
PIE = "Índice cardiotorácico con U-Net frente a una CNN de caja negra · Examen Final · Grupo 6"
AL = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
AN = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}


# ----------------------------------------------------------------------------------------------- texto con marcado
def runs(p, texto, size, color, bold=False, font=None):
    if "\n" in texto:                                   # salto de línea dentro del mismo párrafo
        for k, parte in enumerate(texto.split("\n")):
            if k:
                p.add_line_break()
            runs(p, parte, size, color, bold, font)
        return
    for trozo in re.split(r"(\*\*.+?\*\*|\[\[[mcgsi]\|.+?\]\]|//.+?//)", texto):
        if not trozo:
            continue
        b, c, it, t = bold, color, False, trozo
        if trozo.startswith("**"):
            b, t = True, trozo[2:-2]
        elif trozo.startswith("[["):
            k, t = trozo[2], trozo[4:-2]
            if k == "i":
                it = True
            else:
                c = COL[k]
                b = k != "s" or bold
        elif trozo.startswith("//"):
            it, t = True, trozo[2:-2]
        r = p.add_run()
        r.text = t
        f = r.font
        f.size, f.bold, f.italic = Pt(size), b, it
        f.color.rgb = c
        if font:
            f.name = font


def vineta(p, numero=False):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.26)))
    pPr.set("indent", str(-Inches(0.26)))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", "7A1518")
    if numero:
        fnt = etree.SubElement(pPr, qn("a:buFont"))
        fnt.set("typeface", "+mj-lt")
        etree.SubElement(pPr, qn("a:buAutoNum")).set("type", "arabicPeriod")
    else:
        etree.SubElement(pPr, qn("a:buSzPct")).set("val", "80000")
        fnt = etree.SubElement(pPr, qn("a:buFont"))
        fnt.set("typeface", "Arial")
        etree.SubElement(pPr, qn("a:buChar")).set("char", "▪")


def llenar(tf, paras, size=16, color=TINTA, align="l", anchor="t", font=None, after=5, margin=0.05, bold=False):
    tf.word_wrap = True
    tf.vertical_anchor = AN[anchor]
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin * 0.6)
    for i, pa in enumerate(paras):
        if isinstance(pa, str):
            pa = {"t": pa}
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = AL[pa.get("a", align)]
        p.space_after = Pt(pa.get("after", after))
        if pa.get("b"):
            vineta(p)
        if pa.get("n"):
            vineta(p, numero=True)
        runs(p, pa["t"], pa.get("size", size), pa.get("color", color), pa.get("bold", bold), pa.get("font", font))


def texto(s, x, y, w, h, paras, nombre=None, **kw):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if nombre:
        tb.name = nombre
    llenar(tb.text_frame, paras, **kw)
    return tb


def caja(s, x, y, w, h, paras, borde=LINEA, fondo=FONDO, grosor=1.0, forma=MSO_SHAPE.ROUNDED_RECTANGLE, radio=0.08,
         nombre=None, **kw):
    sh = s.shapes.add_shape(forma, Inches(x), Inches(y), Inches(w), Inches(h))
    if forma == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radio
    if fondo is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fondo
    if borde is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = borde
        sh.line.width = Pt(grosor)
    sh.shadow.inherit = False
    if nombre:
        sh.name = nombre
    kw.setdefault("align", "c")
    kw.setdefault("anchor", "m")
    kw.setdefault("size", 12)
    kw.setdefault("after", 2)
    llenar(sh.text_frame, paras, **kw)
    return sh


def flecha(s, x1, y1, x2, y2, color=TINTA2, punteada=False, grosor=1.25, punta=True):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(grosor)
    if punteada:
        c.line.dash_style = MSO_LINE.DASH
    if punta:
        ln = c.line._get_or_add_ln()
        cola = etree.SubElement(ln, qn("a:tailEnd"))
        cola.set("type", "triangle")
        cola.set("w", "med")
        cola.set("len", "med")
    return c


def imagen(s, ruta, x, y, w=None, h=None, centrar=True):
    """Coloca la imagen dentro de la caja (x, y, w, h) sin deformarla."""
    iw, ih = Image.open(ruta).size
    if w and h:
        e = min(w / iw, h / ih)
        ww, hh = iw * e, ih * e
        if centrar:
            x, y = x + (w - ww) / 2, y + (h - hh) / 2
    elif w:
        ww, hh = w, w * ih / iw
    else:
        hh, ww = h, h * iw / ih
    return s.shapes.add_picture(str(ruta), Inches(x), Inches(y), Inches(ww), Inches(hh))


def borde_celda(celda, color="D8DEE4", ancho=0.75, lados=("B",)):
    tcPr = celda._tc.get_or_add_tcPr()
    for i, lado in enumerate(("L", "R", "T", "B")):
        ln = etree.Element(qn(f"a:ln{lado}"))
        if lado in lados:
            ln.set("w", str(int(Pt(ancho))))
            sf = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
        else:
            ln.set("w", "0")
            etree.SubElement(ln, qn("a:noFill"))
        tcPr.insert(i, ln)


def tabla(s, x, y, anchos, filas, size=12, alto=0.36, cab_size=None, nombre=None, alinear=None):
    n, m = len(filas), len(anchos)
    sh = s.shapes.add_table(n, m, Inches(x), Inches(y), Inches(sum(anchos)), Inches(alto * n))
    if nombre:
        sh.name = nombre
    t = sh.table
    t.first_row = t.horz_banding = False
    for j, a in enumerate(anchos):
        t.columns[j].width = Inches(a)
    for i, fila in enumerate(filas):
        t.rows[i].height = Inches(alto)
        for j, val in enumerate(fila):
            cel = t.cell(i, j)
            cel.fill.solid()
            cel.fill.fore_color.rgb = FONDO if i == 0 else BLANCO
            borde_celda(cel, "7A1518" if i in (0, n - 1) else "D8DEE4", 1.0 if i in (0, n - 1) else 0.5,
                        ("B",) if i else ("T", "B"))
            cel.margin_left = cel.margin_right = Inches(0.06)
            cel.margin_top = cel.margin_bottom = Inches(0.03)
            al = alinear[j] if alinear else "l"
            tf = cel.text_frame
            tf.word_wrap = True
            cel.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.alignment = AL[al]
            runs(p, val, (cab_size or size) if i == 0 else size, GR if i == 0 else TINTA, bold=(i == 0))
    return sh


# --------------------------------------------------------------------------------------------------- estructura
prs = Presentation(str(SL / "plantilla_uni.pptx"))
LAY = next(l for l in prs.slide_layouts if l.name == "CONTENIDO")
for sh in LAY.shapes:                                   # pie de la plantilla (venía del parcial de YOLO)
    if sh.has_text_frame and sh.text_frame.text.startswith("Detecci"):
        p = sh.text_frame.paragraphs[0]
        p.runs[0].text = PIE
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
NUM = next(sh for sh in LAY.placeholders if sh.placeholder_format.type is not None and "Slide Number" in sh.name)

lista = prs.slides._sldIdLst                             # deja solo la portada de la plantilla
for sid in list(lista)[1:]:
    prs.part.drop_rel(sid.rId)
    lista.remove(sid)

notas = {}
for linea in (SL / "GUION.md").read_text(encoding="utf-8").splitlines():
    m = re.match(r"\|\s*(\d+)\s*\|.*\|\s*([^|]+?)\s*\|\s*$", linea)
    if m:
        notas[int(m.group(1))] = m.group(2)


CAJA_NOTAS = copy.deepcopy(prs.slides[0].notes_slide.notes_placeholder._element)   # el maestro de notas no la trae


def poner_notas(diapo, txt):
    ns = diapo.notes_slide
    if ns.notes_text_frame is None:
        ns.shapes._spTree.append(copy.deepcopy(CAJA_NOTAS))
    ns.notes_text_frame.text = txt


def nueva(titulo, subtitulo):
    s = prs.slides.add_slide(LAY)
    s.shapes._spTree.append(copy.deepcopy(NUM._element))
    texto(s, 3.6, 0.18, 9.4, 0.78, [{"t": titulo, "a": "c"}], nombre="Título", size=27, bold=True, anchor="m",
          after=0)
    texto(s, 0.6, 1.13, 12.13, 0.42, [{"t": subtitulo, "a": "c"}], nombre="Subtítulo", size=17, color=GR,
          bold=True, font="Cambria", anchor="m", after=0)
    k = len(prs.slides)
    if k in notas:
        poner_notas(s, notas[k])
    return s


def nota(s, x, y, w, t, size=11, h=0.35, a="l"):
    return texto(s, x, y, w, h, [{"t": t, "a": a}], size=size, color=TINTA2, after=0)


# ------------------------------------------------------------------------------------------------ 1 · portada
por = prs.slides[0]
for sh in por.shapes:
    if not sh.has_text_frame:
        continue
    tf = sh.text_frame
    if sh.name == "Text 1":
        tf.paragraphs[0].runs[0].text = ("Medir como el radiólogo: índice cardiotorácico con U-Net frente a una CNN "
                                         "de caja negra para detectar cardiomegalia")
    elif sh.name == "Text 2":
        tf.paragraphs[0].runs[0].text = "Examen Final · Proyecto del curso"
    elif sh.name == "Text 3":
        tf.paragraphs[0].runs[1].text = ("Redes Neuronales y Aprendizaje Profundo (MIA-203) · Maestría en "
                                         "Inteligencia Artificial")
        tf.paragraphs[2].runs[0].text = "Contreras Avendaño, Juan Sebastian"
    elif sh.name == "Text 4":
        tf.paragraphs[0].runs[0].text = "Noviembre - 2026"
texto(por, 8.2, 6.15, 4.4, 0.35, [{"t": "Repositorio: github.com/loupv99/corazon-v2", "a": "r"}], size=12,
      color=TINTA2)
poner_notas(por, notas.get(1, ""))

# ------------------------------------------------------------------------------- 2 · contexto y antecedentes
s = nueva("Contexto y antecedentes", "Del ICT de Danzer (1919) a las máscaras automáticas de CheXmask (2024)")
y0 = 2.35
flecha(s, 0.7, y0, 12.65, y0, color=LINEA, grosor=4, punta=False)
hitos = [("1919", MED, "**Danzer** define el ICT: si pasa de 0.50, el corazón es sospechoso. Se mide a mano."),
         ("2006", TINTA2, "**SCR**: personas trazan pulmones y corazón en 247 placas JSRT."),
         ("2015–16", TINTA2, "**U-Net** segmenta con pocos datos; **ResNet** entrena redes muy profundas."),
         ("2017", CLA, "**ChestX-ray8/14**: 108 948 placas con etiquetas sacadas del informe."),
         ("2020", MED, "**Sogancioglu**: medir gana a clasificar (AUC 0.977 vs. 0.941) con anotación humana."),
         ("2024", MED, "**CheXmask**: 657 566 máscaras automáticas (HybridGNet).")]
for i, (anio, col, txt) in enumerate(hitos):
    cx = 1.25 + i * 2.15
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - 0.1), Inches(y0 - 0.1), Inches(0.2), Inches(0.2))
    d.fill.solid(); d.fill.fore_color.rgb = col; d.line.fill.background()
    texto(s, cx - 1.0, y0 - 0.62, 2.0, 0.42, [{"t": anio, "a": "c"}], size=17, color=col, bold=True, anchor="b",
          after=0)
    texto(s, cx - 1.02, y0 + 0.2, 2.04, 1.5, [{"t": txt, "a": "c"}], size=12, after=0)
caja(s, 0.9, 4.25, 11.55, 2.1, [], fondo=FONDO, borde=None, nombre="Perú")
texto(s, 0.9, 4.32, 11.55, 0.36, [{"t": "**Y en el Perú**", "a": "c"}], size=15, after=0)
for i, (num, lab) in enumerate([("495", "radiólogos para más de 35 millones"),
                                ("54.8 %", "de radiólogos y tecnólogos, en Lima"),
                                ("3 072", "establecimientos en la Red de Telesalud")]):
    cx = 1.2 + i * 3.75
    texto(s, cx, 4.7, 3.5, 0.62, [{"t": num, "a": "c"}], size=32, color=MED, bold=True, after=0, anchor="m")
    texto(s, cx, 5.32, 3.5, 0.4, [{"t": lab, "a": "c"}], size=13, after=0)
texto(s, 0.9, 5.82, 11.55, 0.4, [{"t": "**La placa ya puede viajar; lo que falta es quien la lea.**", "a": "c"}],
      size=15, after=0)
nota(s, 0.6, 6.47, 12.1, "Fuentes: Danzer 1919 · Wang et al. 2017 · Sogancioglu et al. 2020 · Gaggion et al. 2024 · "
     "MINSA, vía Panamericana (2025, 2026)", size=10, a="c")

# ---------------------------------------------------------------------------------------- 3 · el problema
s = nueva("Planteamiento del problema", "La cardiomegalia se sospecha con una medida que hoy depende del radiólogo")
texto(s, 0.7, 1.85, 7.3, 4.8, [
    {"t": "La cardiomegalia se sospecha con el **índice cardiotorácico**:", "b": 1},
    {"t": "ICT = ancho del corazón ÷ ancho interno del tórax", "a": "c", "size": 18, "font": "Cambria",
     "after": 10},
    {"t": "**ICT > 0.50** en placa PA sugiere cardiomegalia.", "b": 1},
    {"t": "Es una //medida//: el radiólogo traza líneas y divide. Sin radiólogo, la placa espera.", "b": 1},
    {"t": "Dos caminos con aprendizaje profundo: [[m|medir]] el ICT con contornos o [[c|clasificar]] la placa "
          "entera.", "b": 1}], size=18, after=10)
imagen(s, RAIZ / "app/ejemplos/con_cardiomegalia.png", 8.5, 1.8, h=4.45)
nota(s, 8.5, 6.3, 4.45, "Placa de prueba de ChestX-ray14 (NIH)", a="c")

# ------------------------------------------------------------------- 4 · pregunta, hipótesis y contribuciones
s = nueva("Pregunta, hipótesis y contribuciones", "Medir como el radiólogo frente a clasificar sin explicar")
caja(s, 0.9, 1.8, 11.55, 1.25, [
    {"t": "¿Una red que [[m|mide]] como el radiólogo detecta tan bien como una que [[c|clasifica]] sin explicar? "
          "¿Y cuál falla mejor?", "size": 18},
    {"t": "**Hipótesis:** medir detecta al menos igual, y sus errores se pueden ver y corregir.", "size": 15}],
     borde=None, nombre="Pregunta")
cartas = [(MED, "1", "Medir frente a clasificar, en igualdad",
           "Mismas placas, partición y aumento; 3 semillas. U-Net con //máscaras automáticas//, evaluada con "
           "//contornos humanos//.", "→ E1, E2, E3"),
          (CLA, "2", "El atajo de la vista AP",
           "¿Las dos redes usan la proyección en vez del corazón? Corrección explícita para el ICT; Grad-CAM para la "
           "caja negra.", "→ E4, E6"),
          (TINTA2, "3", "Datos y aplicación propios",
           "300 fotos con celular de 100 placas, con marcas ArUco para enderezarlas. App web que mide en el teléfono "
           "y deja corregir.", "→ E5")]
for i, (col, n, tit, cuerpo, exp) in enumerate(cartas):
    x = 0.9 + i * 3.95
    caja(s, x, 3.6, 3.65, 2.55, [{"t": f"**{tit}**", "size": 16, "after": 6}, {"t": cuerpo, "size": 14, "after": 8},
                                 {"t": exp, "size": 14, "bold": True, "color": col}],
         borde=col, fondo=BLANCO, grosor=1.5, align="l", anchor="t", margin=0.18, nombre=f"Contribución {n}")
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - 0.17), Inches(3.43), Inches(0.38), Inches(0.38))
    c.fill.solid(); c.fill.fore_color.rgb = col; c.line.fill.background()
    llenar(c.text_frame, [{"t": f"**{n}**", "a": "c"}], size=13, color=BLANCO, anchor="m", after=0, margin=0)

# ------------------------------------------------------------------------------- 5–6 · trabajos relacionados
CAB = ["Trabajo", "Metodología", "Principales hallazgos", "Aporte al conocimiento", "En qué nos diferenciamos"]
ANCH = [1.9, 2.65, 2.65, 2.25, 2.7]
s = nueva("Trabajos relacionados (1/2)", "Arquitecturas, explicación visual y datos a gran escala")
tabla(s, 0.6, 1.85, ANCH, [CAB,
      ["**Ronneberger, 2015**\nU-Net", "Codificador-decodificador con saltos; aumento de datos fuerte",
       "Segmenta bien con muy pocas imágenes; ganó ISBI 2015", "Segmentación precisa con pocos datos",
       "[[m|La usamos para medir el ICT]], entrenada con máscaras automáticas"],
      ["**He, 2016**\nResNet", "Bloques residuales y = F(x) + x, hasta 152 capas",
       "3.57 % de error en ImageNet; 1.er puesto en ILSVRC 2015", "Redes muy profundas que sí se optimizan",
       "ResNet-50 como [[c|caja negra de referencia]]"],
      ["**Selvaraju, 2017**\nGrad-CAM", "Gradientes sobre la última capa convolucional, sin reentrenar",
       "Localiza regiones relevantes y revela sesgos del conjunto", "Explicación visual para cualquier CNN",
       "[[c|La cuantificamos]] en aciertos y en errores"],
      ["**Wang, 2017**\nChestX-ray8/14", "108 948 placas de 32 717 pacientes; etiquetas del informe con PLN",
       "Detecta con etiquetas débiles, pero leer la placa sigue siendo difícil", "Primera base pública grande de tórax",
       "Su etiqueta no equivale a ICT > 0.50 (esp. 0.504)"]], size=13, alto=0.95, cab_size=13,
      nombre="Tabla trabajos 1")
s = nueva("Trabajos relacionados (2/2)", "Cardiomegalia, máscaras anatómicas y robustez")
tabla(s, 0.6, 1.85, ANCH, [CAB,
      ["**Sogancioglu, 2020**", "Segmentar (778 anotadas) vs. clasificar (65 000)",
       "AUC 0.977 vs. 0.941 (367 placas, 2 radiólogos)", "Medir gana con 100 veces menos anotaciones",
       "Etiqueta del informe y [[m|máscaras automáticas]]: gana clasificar"],
      ["**Chamveha, 2020**", "U-Net + VGG16; ICT = (MRD + MLD) / ID", "76.5 % de medidas aceptadas sin cambios",
       "ICT automático usable en el informe", "[[m|MRD y MLD por separado]] (E6); vista AP y fotos"],
      ["**Gaggion, 2023–24**\nHybridGNet, CheXmask", "Conv. + grafos; 657 566 máscaras",
       "Contornos plausibles donde otros modelos fallan", "Máscaras anatómicas a gran escala",
       "Las [[m|destilamos en una U-Net]] para el navegador"],
      ["**Zech, 2018; Phillips, 2020**", "CNN en 3 hospitales (158 323 placas); fotos con celular",
       "Reconoce el hospital (99.95 %, NIH); CheXphoto: robustez", "Atajos del origen; escenario de fotos",
       "Atajo de la [[c|vista AP]] en ambos enfoques; fotos propias con ArUco"]], size=13, alto=0.95,
      cab_size=13, nombre="Tabla trabajos 2")

# ------------------------------------------------------------------------------------- 7 · fundamento teórico
s = nueva("Fundamento teórico", "Una red que localiza bordes frente a una que resume toda la placa")
texto(s, 0.6, 1.75, 6.0, 0.4, ["[[m|U-Net — medir]]"], size=17, after=0)
for i, ch in enumerate((32, 64, 128, 256)):
    w = 0.75 + 0.18 * i
    y = 2.45 + 0.45 * i
    caja(s, 0.85 + 0.3 * i, y, w, 0.32, [f"{ch}"], borde=MED, fondo=RGBColor(0xE3, 0xEE, 0xFA), size=11,
         forma=MSO_SHAPE.RECTANGLE, margin=0)
    caja(s, 5.35 - 0.3 * i - w, y, w, 0.32, [f"{ch}"], borde=MED, fondo=RGBColor(0xE3, 0xEE, 0xFA), size=11,
         forma=MSO_SHAPE.RECTANGLE, margin=0)
    flecha(s, 0.85 + 0.3 * i + w, y + 0.16, 5.35 - 0.3 * i - w, y + 0.16, color=MED, punteada=True, grosor=1)
caja(s, 2.65, 4.3, 0.95, 0.32, ["512"], borde=TINTA, fondo=FONDO, size=11, forma=MSO_SHAPE.RECTANGLE, margin=0)
nota(s, 0.6, 2.1, 1.9, "placa 256 px", size=11)
nota(s, 3.8, 2.1, 1.9, "3 sigmoides", size=11, a="r")
texto(s, 0.6, 4.7, 5.4, 0.62, [{"t": "cuello de 16 px: contexto de toda la placa", "a": "c"},
                               {"t": "[[m|- - - →]] saltos: devuelven el detalle del borde", "a": "c"}],
      size=11, color=TINTA2, after=0)
texto(s, 0.6, 5.4, 6.0, 1.4, [
    {"t": "El ICT depende del borde: 3 px de error en 256 cambian el ICT en ~0.01.", "b": 1},
    {"t": "Salida **verificable**: contornos que el médico ve y corrige.", "b": 1}], size=15, after=4)
texto(s, 6.9, 1.75, 6.0, 0.4, ["[[c|ResNet-50 — clasificar]]"], size=17, after=0)
xs = [7.0, 7.55, 8.95, 10.35, 11.85, 12.45]
yb = 3.2
texto(s, xs[0], yb - 0.17, 0.4, 0.34, ["x"], size=14, font="Cambria", after=0, anchor="m")
for k, et in enumerate(("conv 1×1", "conv 3×3", "conv 1×1")):
    caja(s, xs[k + 1], yb - 0.2, 1.2, 0.4, [et], borde=CLA, fondo=RGBColor(0xFD, 0xEC, 0xE4), size=11,
         forma=MSO_SHAPE.RECTANGLE, margin=0)
mas = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(xs[4]), Inches(yb - 0.15), Inches(0.3), Inches(0.3))
mas.fill.solid(); mas.fill.fore_color.rgb = BLANCO; mas.line.color.rgb = TINTA
llenar(mas.text_frame, [{"t": "+", "a": "c"}], size=12, anchor="m", after=0, margin=0)
texto(s, xs[5], yb - 0.17, 0.4, 0.34, ["y"], size=14, font="Cambria", after=0, anchor="m")
for a, b in ((7.3, 7.55), (8.75, 8.95), (10.15, 10.35), (11.55, 11.85), (12.15, 12.45)):
    flecha(s, a, yb, b, yb, grosor=1)
flecha(s, 7.15, yb - 0.2, 7.15, yb - 0.55, color=CLA, grosor=1.5, punta=False)
flecha(s, 7.15, yb - 0.55, 12.0, yb - 0.55, color=CLA, grosor=1.5, punta=False)
flecha(s, 12.0, yb - 0.55, 12.0, yb - 0.15, color=CLA, grosor=1.5)
texto(s, 8.4, yb - 0.9, 3.0, 0.32, [{"t": "identidad: y = F(x) + x", "a": "c"}], size=12, color=CLA, after=0)
caja(s, 7.4, 3.85, 4.9, 1.05, [{"t": "**Grad-CAM** sobre la última capa:", "size": 13},
                              {"t": "L = ReLU(Σₖ αₖ Aᵏ),   αₖ = (1/Z) Σᵢⱼ ∂y/∂Aᵏᵢⱼ", "size": 14, "font": "Cambria"}],
     borde=CLA, fondo=BLANCO, grosor=1.25)
texto(s, 6.9, 5.1, 6.0, 1.7, [
    {"t": "Preentrenada en ImageNet: buen punto de partida con 5 416 placas.", "b": 1},
    {"t": "Ve toda la placa: puede usar señales que el ICT ignora… y atajos.", "b": 1}], size=15, after=4)

# --------------------------------------------------------------------------------- 8 · preparación de datos
s = nueva("Preparación de datos", "Cuatro fuentes llevadas a un mismo formato de 256 px")
filas = [(CLA, "ChestX-ray14", "NIH · 112 120 placas",
          ["2 776 con cardiomegalia + 6 000 sin ella", "Partición por **paciente**; prueba oficial del NIH",
           "PNG → gris → 256 px (promedio de área)", "Guarda mm/píxel y vista PA/AP"], "**8 776** placas"),
         (MED, "CheXmask", "máscaras débiles",
          ["CSV de 2 GB leído por partes", "RLE → 3 máscaras de 1 024 px", "Reducción por área + umbral 0.5",
           "Calidad RCA e ICT sin red"], "**8 666** con RCA ≥ 0.7"),
         (MED, "SCR / JSRT", "contornos humanos",
          ["247 contornos trazados a mano", "Rasterizar a máscaras",
           "Imágenes JSRT **limpias** (sin contorno dibujado)", "Gris invertido si hace falta, comprobado"],
          "**100** ajuste · **147** prueba"),
         (TINTA2, "Fotos propias", "E5",
          ["100 placas: 25 por grupo PA/AP × con/sin", "Monitor con 5 marcas ArUco",
           "iPhone de frente e inclinada; Honor de frente", "Homografía → 256 px"], "**300** fotos")]
for i, (col, nom, sub, pasos, fin) in enumerate(filas):
    y = 1.85 + i * 1.17
    texto(s, 0.55, y, 1.75, 0.9, [{"t": nom, "bold": True, "color": col, "size": 14, "after": 0},
                                  {"t": sub, "color": TINTA2, "size": 11}], anchor="m", after=0)
    for k, ps in enumerate(pasos):
        x = 2.35 + k * 2.2
        caja(s, x, y, 1.95, 0.9, [ps], borde=LINEA, fondo=FONDO, size=11, radio=0.06)
        flecha(s, x + 1.95, y + 0.45, x + 2.2, y + 0.45, grosor=1)
    caja(s, 11.15, y + 0.05, 1.6, 0.8, [fin], borde=col, fondo=BLANCO, size=12, grosor=1.5, radio=0.06)
nota(s, 0.6, 6.55, 12.1, "Salida común: arrays de 256 px (pulmón derecho, izquierdo, corazón) + una tabla por placa. "
     "Una sola vez, ~20 min en Kaggle.", a="c")

# -------------------------------------------------------------------------------------------------- 9 · datos
s = nueva("Datos", "8 776 placas con partición por paciente; la vista está asociada a la etiqueta")
tabla(s, 0.6, 1.85, [2.6, 1.15, 1.15, 1.15], [["ChestX-ray14", "Placas", "Cardio.", "AP"],
      ["Entrenamiento", "5 416", "1 471", "1 829"], ["Validación", "1 006", "236", "357"],
      ["Prueba (oficial NIH)", "2 354", "1 069", "1 335"]], size=14, alto=0.42, alinear="lrrr", nombre="Tabla datos")
texto(s, 0.6, 3.75, 6.1, 3.0, [
    {"t": "Partición por **paciente** (5 674): nadie aparece en dos particiones.", "b": 1},
    {"t": "Negativos con y sin otros hallazgos: la red no aprende «sano vs. enfermo».", "b": 1},
    {"t": "[[c|La vista está asociada a la etiqueta]]: 43.7 % AP en cardiomegalia frente a 38.5 % sin ella.",
     "b": 1}], size=15, after=6)
texto(s, 7.05, 1.8, 5.7, 0.4, ["**Qué papel cumple cada fuente**"], size=15, after=0)
roles = [(CLA, "**ChestX-ray14**\netiqueta del informe", "**Diagnóstico**: entrena la ResNet y evalúa a ambos (E2, E4)"),
         (MED, "**CheXmask**\nautomáticas, CC BY 4.0", "**Enseña**: entrena la U-Net (E1a, E1b)"),
         (MED, "**SCR / JSRT**\nhumanas, CC BY 4.0", "**Juzga y ajusta**: Dice e ICC (E1, E3); 80 + 20 para E1c"),
         (TINTA2, "**Fotos propias**\n2 celulares", "**Pone a prueba**: decisiones que cambian (E5)")]
for i, (col, f, r) in enumerate(roles):
    y = 2.3 + i * 1.12
    a, b = f.split("\n")
    caja(s, 7.05, y, 2.15, 0.85, [{"t": a, "a": "l"}, {"t": b, "a": "l", "size": 11, "color": TINTA2}], borde=col,
         fondo=BLANCO, grosor=1.25, size=12, margin=0.1)
    flecha(s, 9.2, y + 0.42, 9.5, y + 0.42, color=col, grosor=1.5)
    caja(s, 9.5, y, 3.25, 0.85, [{"t": r, "a": "l"}], borde=None, fondo=FONDO, size=12, margin=0.1)

# --------------------------------------------------------------------- 10 · etiquetas débiles y referencia humana
s = nueva("Etiquetas débiles y referencia humana",
          "La U-Net aprende de máscaras automáticas y se evalúa contra contornos humanos")
imagen(s, RAIZ / "results/kaggle-v1/control_scr.png", 1.9, 1.75, w=9.5)
nota(s, 0.6, 4.18, 12.1, "Contornos humanos de SCR rasterizados (azul: pulmón derecho; verde: izquierdo; rojo: corazón)",
     a="c")
cadena = [(TINTA2, "**HybridGNet**\nmaestro: conv. + grafos"), (MED, "**CheXmask**\n8 666 máscaras, RCA ≥ 0.7"),
          (MED, "**U-Net**\nalumna: 7.8 M parám."), (TINTA, "**SCR**: 147 placas\njuez humano (Dice, ICC)")]
for i, (col, t) in enumerate(cadena):
    x = 1.1 + i * 2.95
    a, b = t.split("\n")
    caja(s, x, 4.65, 2.45, 0.8, [a, {"t": b, "size": 12}], borde=col, fondo=BLANCO, grosor=1.25, size=13)
    if i < 3:
        flecha(s, x + 2.45, 5.05, x + 2.95, 5.05, grosor=1.5)
flecha(s, 10.0, 5.45, 10.0, 5.75, color=MED, punteada=True, grosor=1.25, punta=False)
flecha(s, 10.0, 5.75, 7.27, 5.75, color=MED, punteada=True, grosor=1.25, punta=False)
flecha(s, 7.27, 5.75, 7.27, 5.47, color=MED, punteada=True, grosor=1.25)
texto(s, 7.4, 5.72, 2.6, 0.32, [{"t": "E1c: 80 + 20 contornos humanos", "a": "c"}], size=11, color=MED, after=0)
nota(s, 0.6, 6.15, 12.1, "Una destilación hacia una red que cabe en un navegador. Sin volteo horizontal en el "
     "aumento: el corazón está a la izquierda del paciente.", a="c")

# -------------------------------------------------------------------------------- 11 · arquitectura del sistema
s = nueva("Arquitectura del sistema", "Dos caminos para la misma radiografía")
caja(s, 0.8, 2.85, 2.3, 0.95, ["Radiografía", "1 × 256 × 256"], size=14)
for fila, (col, cajas) in enumerate([(MED, [("**U-Net**", "7.8 M parámetros, desde cero"),
                                            ("3 máscaras", "pulmones D e I, corazón"),
                                            ("MRD, MLD, ID", "⇒ **ICT**")]),
                                     (CLA, [("**ResNet-50**", "23.5 M parám., preentrenada"),
                                            ("p(cardiomegalia)", ""), ("**Grad-CAM**", "sobre layer4")])]):
    y = 1.95 + fila * 1.85
    flecha(s, 3.1, 3.32, 3.6, y + 0.47, grosor=1.5)
    for k, (a, b) in enumerate(cajas):
        x = 3.6 + k * 3.1
        caja(s, x, y, 2.6, 0.95, [a] + ([b] if b else []), borde=col if k != 1 else LINEA, size=14, grosor=1.5)
        if k < 2:
            flecha(s, x + 2.6, y + 0.47, x + 3.1, y + 0.47, grosor=1.5)
texto(s, 0.7, 4.95, 5.9, 1.8, ["[[m|U-Net]]: 4 niveles (32–256 canales) + cuello de 512; cada bloque = 2 × (conv 3×3, "
                               "BatchNorm, ReLU); sube con conv. transpuesta y concatena el salto; salida 1×1 con 3 "
                               "sigmoides."], size=14)
texto(s, 6.9, 4.95, 5.8, 1.8, ["[[c|ResNet-50]]: capa final cambiada por una salida; gris repetido en 3 canales; "
                               "mismas imágenes, partición y aumento que la U-Net."], size=14)

# ----------------------------------------------------------------------------------------------- 12 · formulación
s = nueva("Formulación", "De la pérdida de segmentación al índice cardiotorácico")
caja(s, 0.6, 1.85, 6.1, 1.75, [
    {"t": "[[m|Segmentación]]: BCE + Dice promediado por canal", "a": "l", "size": 14, "after": 6},
    {"t": "L_seg = BCE(ŷ, y) + 1 − (1/3) Σ_c [ (2 Σᵢ ŷ_c,i · y_c,i + ε) / (Σᵢ ŷ_c,i + Σᵢ y_c,i + ε) ]",
     "font": "Cambria", "size": 12, "after": 6},
    {"t": "Promediar por canal evita que los pulmones, más grandes, dominen al corazón.", "a": "l", "size": 12,
     "color": TINTA2}], borde=MED, fondo=BLANCO, grosor=1.25, anchor="t", margin=0.15)
caja(s, 0.6, 3.85, 6.1, 1.6, [
    {"t": "[[c|Clasificación]]: BCE con peso w₊ = N₋ / N₊", "a": "l", "size": 14, "after": 6},
    {"t": "L_cls = −(1/N) Σₙ [ w₊ yₙ log pₙ + (1 − yₙ) log(1 − pₙ) ]", "font": "Cambria", "size": 14, "after": 6},
    {"t": "El peso compensa que hay menos placas con cardiomegalia.", "a": "l", "size": 12, "color": TINTA2}],
     borde=CLA, fondo=BLANCO, grosor=1.25, anchor="t", margin=0.15)
texto(s, 7.05, 1.8, 5.7, 0.4, ["[[m|Del contorno al número]]"], size=15, after=0)
imagen(s, FIG / "medida_real.jpg", 7.05, 2.2, h=2.75)
texto(s, 9.95, 2.2, 2.85, 2.75, [
    {"t": "Salida real de la U-Net en la placa de ejemplo:", "size": 12, "color": TINTA2, "after": 6},
    {"t": "MRD = 32 px", "size": 14}, {"t": "MLD = 87 px", "size": 14}, {"t": "ID = 228 px", "size": 14},
    {"t": "**ICT = 119 / 228 = 0.52**", "size": 15}], after=3)
pasos = ["Umbral 0.5 + mayor componente", "ID y xₘ en la fila más ancha", "MRD = xₘ − x_min\nMLD = x_max − xₘ",
         "ICT = (MRD + MLD) / ID"]
for k, t in enumerate(pasos):
    x = 7.05 + k * 1.45
    caja(s, x, 5.15, 1.3, 0.95, t.split("\n"), borde=MED if k == 3 else None, fondo=FONDO, size=11, margin=0.04,
         radio=0.06)
    if k < 3:
        flecha(s, x + 1.3, 5.62, x + 1.45, 5.62, grosor=1)
nota(s, 7.05, 6.2, 5.7, "Si falta una estructura, la placa queda como //no medible//.")

# ------------------------------------------------------------------------------ 13 · entrenamiento y costo
s = nueva("Entrenamiento y costo computacional", "Cinco configuraciones, tres semillas y 4.2 h de GPU")
nota(s, 0.6, 1.75, 7.3, "cada caja × 3 semillas (0, 1, 2) ⇒ media ± D. E.", a="c")
conf = [[(MED, "**E1a** U-Net", "todas las débiles"), (MED, "**E1b** U-Net", "solo RCA ≥ 0.7"),
         (MED, "**E1c** ajuste", "+ 80 humanas")],
        [(CLA, "**ResNet-50**", "PA + AP"), (CLA, "**ResNet-50**", "solo PA (E4)"),
         (CLA, "**Grad-CAM**", "de cada ResNet")]]
for f, fila in enumerate(conf):
    for k, (col, a, b) in enumerate(fila):
        x = 0.75 + k * 2.45
        caja(s, x, 2.15 + f * 0.9, 2.1, 0.72, [a, {"t": b, "size": 11}], borde=col, fondo=BLANCO, grosor=1.25, size=13)
    flecha(s, 0.75 + 2.45 + 2.1, 2.51 + f * 0.9, 0.75 + 2 * 2.45, 2.51 + f * 0.9, grosor=1.25)
tabla(s, 0.75, 4.05, [2.1, 1.65, 1.85, 1.6], [["", "U-Net", "Ajuste E1c", "ResNet-50"],
      ["Épocas · lote", "30 · 16", "30 · 8", "12 · 32"], ["Tasa máx.", "10⁻³", "2×10⁻⁴", "3×10⁻⁴"],
      ["Elige época por", "Dice val.", "Dice 20 humanas", "AUC val."]], size=12, alto=0.36, alinear="lccc",
      nombre="Tabla entrenamiento")
nota(s, 0.75, 5.6, 7.2, "AdamW (decaimiento 10⁻⁴), OneCycle, precisión mixta. Aumento: rotación ±10°, escala 0.9–1.1, "
     "traslación ±5 %, brillo/contraste ±15 %.", h=0.6)
texto(s, 8.4, 1.75, 4.4, 0.6, [{"t": "**Minutos de GPU**", "after": 0}, {"t": "Kaggle, 2 × Tesla T4", "size": 11,
                                                                          "color": TINTA2}], size=15, after=0)
cd = CategoryChartData()
cd.categories = ["Día 1: 2 U-Net + ajuste", "Día 2: U-Net, 3 semillas", "Día 2: ResNet + Grad-CAM"]
cd.add_series("Minutos", (62.6, 150.6, 38.2))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(8.3), Inches(2.3), Inches(4.5), Inches(2.1), cd)
gf.name = "Gráfico minutos GPU"
ch = gf.chart
ch.has_legend = False
ch.has_title = False
ch.plots[0].gap_width = 60
ch.plots[0].has_data_labels = True
dl = ch.plots[0].data_labels
dl.font.size, dl.font.bold = Pt(11), True
dl.number_format, dl.number_format_is_linked = "0.0", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
for k, col in enumerate((MED, MED, CLA)):
    pt = ch.plots[0].series[0].points[k]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
ch.category_axis.reverse_order = True
ch.category_axis.tick_labels.font.size = Pt(10)
ch.category_axis.format.line.color.rgb = LINEA
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.maximum_scale = 185
texto(s, 8.4, 4.5, 4.4, 0.75, ["**4.2 h** de GPU"], size=30, after=0)
texto(s, 8.4, 5.3, 4.4, 1.45, ["Umbral de la ResNet: Youden en validación. ICT: **0.50 clínico**, sin ajustar. **La "
                               "prueba no se usó para elegir nada.** E5 y E6, en el CPU de una laptop."], size=12)

# ---------------------------------------------------------------------------------------- 14 · diseño experimental
s = nueva("Diseño experimental", "Seis experimentos, cada uno ligado a una contribución")
tabla(s, 0.6, 1.85, [2.6, 6.0, 3.5], [["Experimento", "Qué se compara", "Métrica"],
      ["**E1** Ablación de etiquetas", "Débiles todas / filtradas RCA ≥ 0.7 / + 80 humanas", "Dice en SCR"],
      ["**E2** Línea base", "ICT U-Net, ResNet-50, ICT de CheXmask sin red, reg. log. [ICT, vista]",
       "AUC, sens., esp., F1"],
      ["**E3** Acuerdo clínico", "ICT de la U-Net vs. ICT de contornos humanos", "ICC(A,1), Bland-Altman"],
      ["**E4** Proyección", "AUC por vista; ResNet solo PA; fuga de vista; Grad-CAM", "AUC PA/AP, fuga"],
      ["**E5** Fotos", "Original vs. 3 condiciones reales + 6 sintéticas", "% de decisiones que cambian, AUC"],
      ["**E6** Lado derecho", "MRD/ID y MLD/ID por separado", "AUC, aciertos vs. errores"]], size=14, alto=0.56,
      nombre="Tabla experimentos")
nota(s, 0.6, 6.05, 12.1, "Fuga de vista = AUC del puntaje para separar AP de PA //solo entre placas sin cardiomegalia//; "
     "0.5 = el puntaje no codifica la vista.", size=12, a="c")

# ---------------------------------------------------------------------------------------------- 15–18 · resultados
s = nueva("Resultado 1: línea base", "La caja negra detecta mejor: AUC 0.892 frente a 0.814")
imagen(s, FIG / "auc_metodos.png", 0.6, 1.8, w=7.4, h=4.95)
texto(s, 8.3, 2.0, 4.45, 4.6, [
    {"t": "[[c|ResNet-50]]: AUC **0.892**; gana en cada semilla y vista.", "b": 1},
    {"t": "[[m|ICT]]: **0.814**; supera al ICT de CheXmask sin red (0.801).", "b": 1},
    {"t": "Al umbral 0.50: sens. 0.918, esp. 0.504.", "b": 1},
    {"t": "El umbral no es el problema: Youden en validación = 0.495–0.497.", "b": 1}], size=16, after=10)

s = nueva("Resultado 2: acuerdo con contornos humanos",
          "Sin un solo contorno humano, la U-Net mide con ICC 0.945–0.951")
tabla(s, 1.3, 1.85, [3.4, 1.6, 1.6, 1.5, 2.2], [
    ["Variante (147 placas SCR)", "Dice corazón", "Dice medio", "Error ICT", "ICC(A,1)"],
    ["E1a débil, todas", "0.936", "0.962", "0.012", "0.948–0.953"],
    ["E1b débil, RCA ≥ 0.7", "0.936", "0.961", "0.012", "0.945–0.951"],
    ["E1c E1b + 80 humanas", "**0.951**", "**0.971**", "**0.010**", "**0.964–0.971**"]], size=15, alto=0.45,
    alinear="lcccc", nombre="Tabla acuerdo")
texto(s, 0.7, 4.15, 5.9, 2.5, [
    {"t": "Filtrar por calidad no cambia nada: solo 110 de 8 776 máscaras bajo RCA 0.7.", "b": 1},
    {"t": "E1c: sesgo de −0.0035 a −0.0005; Bland-Altman ≈ ±0.03.", "b": 1}], size=16, after=8)
caja(s, 6.9, 4.15, 5.85, 1.65, [
    {"t": "[[c|Paradoja]]: E1c mide mejor, pero su AUC baja de 0.814 a **0.796**.", "a": "l", "size": 16,
     "after": 8},
    {"t": "Medir mejor no es diagnosticar mejor cuando la etiqueta no es la medida. Además, la mejora se mide en "
          "JSRT, el mismo dominio del ajuste.", "a": "l", "size": 13, "color": TINTA2}],
     borde=None, fondo=FONDO, anchor="t", margin=0.18)

s = nueva("Resultado 3: el atajo de la proyección AP",
          "Ambos enfoques pierden en AP; solo el ICT se corrige con una fórmula")
imagen(s, FIG / "vista_pa_ap.png", 0.6, 1.8, w=7.3, h=4.95)
texto(s, 8.2, 1.95, 4.6, 4.8, [
    {"t": "Todos pierden en AP. La ResNet que solo vio PA pierde **0.141**.", "b": 1},
    {"t": "La ResNet [[c|codifica la vista]]: fuga 0.646 (0.710 solo PA).", "b": 1},
    {"t": "En el [[m|ICT]] la causa es física (magnificación): esp. AP = 0.356.", "b": 1},
    {"t": "[[m|Corrección con la vista]] (reg. log.): fuga **0.691 → 0.493**, esp. AP **0.356 → 0.619**, a costa de "
          "0.004 de AUC.", "b": 1}], size=15, after=10)

s = nueva("Resultado 4: fotos con celular", "El ICT cambia de decisión en menos placas y solo cerca del umbral")
imagen(s, FIG / "foto_inclinada.jpg", 0.6, 1.85, w=5.0)
nota(s, 0.6, 4.75, 5.0, "iPhone 16 Pro inclinado: marcas ArUco → homografía → recorte.")
texto(s, 0.6, 5.3, 5.0, 0.5, ["AUC: ambos pierden 0.02–0.03."], size=16)
imagen(s, FIG / "fotos_cambios.png", 5.95, 1.8, w=6.8, h=3.55)
texto(s, 6.1, 5.4, 6.65, 1.4, [
    {"t": "El [[m|ICT]] cambia de decisión solo cerca del umbral: mediana a **0.013** de 0.50 (0.056 en todas).",
     "b": 1},
    {"t": "Las degradaciones sintéticas //subestimaron// el efecto real.", "b": 1}], size=15, after=5)

# ---------------------------------------------------------------------------------------------- 19 · la aplicación
s = nueva("La aplicación", "Medir en el celular, sin enviar la imagen")
imagen(s, RAIZ / "paper/figs/app.png", 2.0, 1.75, w=9.33)
texto(s, 0.9, 5.7, 5.7, 1.15, [
    {"t": "U-Net exportada a **ONNX** (31 MB), corre en el navegador.", "b": 1},
    {"t": "Verificada contra PyTorch: dif. máx. 2.5×10⁻⁶; mismo ICT que Python.", "b": 1},
    {"t": "Para fotos: el usuario marca 4 esquinas y se endereza.", "b": 1}], size=13, after=2)
texto(s, 6.85, 5.7, 5.9, 1.15, [
    {"t": "**Corregir bordes**: el médico arrastra los bordes y el ICT cambia.", "b": 1},
    {"t": "**Aviso de calidad** (incertidumbre de la red): fotos avisadas cambian de decisión 20 % vs. 2.9 %.",
     "b": 1},
    {"t": "1.7–3.2 s por medición en una laptop · loupv99.github.io/corazon-v2", "b": 1}], size=13, after=2)

# ------------------------------------------------------------------------------------------------ 20 · Grad-CAM
s = nueva("Interpretabilidad: Grad-CAM", "El mapa mira al corazón, también cuando la red se equivoca")
imagen(s, FIG / "gradcam_4.png", 0.6, 1.8, w=7.2)
nota(s, 0.6, 3.63, 7.2, "ResNet-50, semilla 0; blanco: corazón de CheXmask. En la 2.ª se ve el rótulo «PORTABLE» "
     "(vista AP).")
imagen(s, FIG / "gradcam_grupos.png", 0.6, 4.05, w=7.2, h=2.75)
texto(s, 8.15, 2.0, 4.6, 4.7, [
    {"t": "Energía en el corazón respecto de un mapa uniforme, por grupo (3 semillas × 2 vistas).", "b": 1},
    {"t": "Falsos positivos = verdaderos positivos (5.3–6.2 vs. 5.4–6.2).", "b": 1},
    {"t": "El mapa dice //dónde// mira, no //si acertó//.", "b": 1},
    {"t": "Falsos positivos más frecuentes en AP en las 3 semillas.", "b": 1}], size=16, after=10)

# ------------------------------------------------------------------------------------------------ 21 · discusión
s = nueva("Discusión", "Explicar no es lo mismo que medir")
texto(s, 0.7, 1.8, 5.9, 3.0, [
    {"t": "[[c|Por qué gana la caja negra]]", "size": 19, "after": 8},
    {"t": "Ve toda la placa; el ICT la reduce a un cociente de anchos.", "b": 1},
    {"t": "La etiqueta del NIH sale del texto del informe, no es «ICT > 0.50»: las distribuciones se solapan.",
     "b": 1},
    {"t": "Aprende de la vista (fuga 0.646): parte de su ventaja puede ser atajo.", "b": 1}], size=17, after=8)
texto(s, 6.9, 1.8, 5.85, 3.0, [
    {"t": "[[m|Lo que se gana al medir]]", "size": 19, "after": 8},
    {"t": "Errores con nombre: AP → magnificación → corrección.", "b": 1},
    {"t": "Con fotos cambia menos decisiones y solo en casos limítrofes.", "b": 1},
    {"t": "El error se ve en la placa y se corrige arrastrando una línea.", "b": 1}], size=17, after=8)
caja(s, 0.7, 5.0, 12.05, 1.45, [
    {"t": "**E6: lo que el ICT frontal no ve**", "size": 15, "color": GR, "after": 6},
    {"t": "MRD/ID separa peor que MLD/ID (AUC 0.697 vs. 0.735). Los falsos negativos no esconden un lado derecho "
          "grande: su MRD/ID (0.138–0.154) es como el de los verdaderos negativos (0.137–0.148). Para el ventrículo "
          "derecho haría falta la placa lateral.", "size": 14}], borde=GR, fondo=FONDO, align="l", anchor="t",
     margin=0.18, grosor=1.0, nombre="Bloque E6")

# ------------------------------------------------------------------------------------- 22 · limitaciones e impacto
s = nueva("Limitaciones, sesgos e impacto", "Una herramienta de apoyo, no un diagnóstico")
texto(s, 0.7, 1.8, 5.95, 4.9, [
    {"t": "**Limitaciones y sesgos**", "size": 20, "after": 10},
    {"t": "Etiquetas del NIH extraídas por procesamiento de lenguaje: con ruido.", "b": 1},
    {"t": "Población de EE. UU. y Japón; sin placas peruanas. Umbral 0.50 de poblaciones de llanura.", "b": 1},
    {"t": "Fotos de un monitor, no de placas físicas; una sesión, un operador.", "b": 1},
    {"t": "Referencia dorada: 147 placas JSRT, todas PA.", "b": 1}], size=18, after=12)
texto(s, 6.95, 1.8, 5.8, 4.9, [
    {"t": "**Impacto y uso responsable**", "size": 20, "after": 10},
    {"t": "Apoyo para una segunda lectura, **no un diagnóstico** ni un dispositivo médico.", "b": 1},
    {"t": "La imagen no sale del teléfono.", "b": 1},
    {"t": "Riesgos: aplicar 0.50 a placas AP sin corregir; sesgo de automatización con la caja negra; usarlo en "
          "poblaciones no evaluadas.", "b": 1},
    {"t": "Costo: 4.2 h de GPU T4 para todo el proyecto.", "b": 1}], size=18, after=12)

# ----------------------------------------------------------------------------------------------- 23 · conclusiones
s = nueva("Conclusiones y trabajo futuro", "Lo valioso es qué error puede revisar quien usa la herramienta")
texto(s, 0.7, 1.8, 6.6, 5.0, [
    {"t": "Una U-Net entrenada solo con máscaras automáticas mide el ICT con ICC [[m|0.945–0.951]] ([[m|0.964–0.971]] "
          "con 80 contornos humanos).", "b": 1},
    {"t": "La [[c|ResNet-50]] detecta mejor ([[c|0.892]] vs. [[m|0.814]]), pero codifica la vista y cambia más "
          "decisiones con fotos.", "b": 1},
    {"t": "El ICT falla de forma **predecible y corregible**.", "b": 1},
    {"t": "Sin radiólogo, la elección no es solo de exactitud: es qué error puede revisar quien usa la herramienta.",
     "b": 1}], size=19, after=14)
texto(s, 7.6, 1.8, 5.15, 4.3, [
    {"t": "**Trabajo futuro**", "size": 20, "after": 10},
    {"t": "Validar con placas peruanas e ICT medido por radiólogos; umbral por vista y por altitud.", "n": 1},
    {"t": "Combinar: ResNet como alerta + ICT como explicación revisable.", "n": 1},
    {"t": "Esquinas automáticas y fotos de placas físicas.", "n": 1},
    {"t": "Comparar con HybridGNet y modelos de base tipo Transformer.", "n": 1}], size=17, after=10)
nota(s, 7.6, 6.2, 5.15, "Código, resultados y app: github.com/loupv99/corazon-v2 · loupv99.github.io/corazon-v2")

salida = SL / "slides.pptx"
prs.save(str(salida))
print("guardado:", salida, "·", len(prs.slides), "diapositivas")
