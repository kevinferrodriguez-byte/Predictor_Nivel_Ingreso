# -*- coding: utf-8 -*-
"""
Parcial práctico 1 — Machine Learning: inferencia con Streamlit
Predictor de nivel de ingreso (>50K USD/año) — Dataset Adult / Census Income (UCI)
Maestría en Inteligencia Artificial y Ciencia de Datos — Universidad Autónoma de Occidente

Ejecutar desde la carpeta del proyecto (donde está este archivo):
    streamlit run app.py

La app NO entrena modelos: carga los Pipelines (preprocesamiento + modelo) que guardó el
notebook en modelos/ y solo hace inferencia (predict / predict_proba).
"""
import base64
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================================ rutas
BASE_DIR = Path(__file__).resolve().parent
DIR_MODELOS = BASE_DIR / "modelos"
DIR_IMG = BASE_DIR / "assets" / "img"
DIR_FUENTES = BASE_DIR / "assets" / "fonts"
RUTA_METADATOS = DIR_MODELOS / "metadatos.json"

# ============================================================== paleta institucional
ROJO, ROJO_OSCURO, CARBON = "#E30613", "#A3000D", "#1D1D1F"
TEAL, LADRILLO, GRIS = "#3E9C8F", "#C8553D", "#6B7280"

# ======================================================= textos de la interfaz (ES)
# El modelo recibe SIEMPRE los valores originales del dataset (en inglés);
# las traducciones solo cambian lo que ve la persona usuaria.
TRADUCCIONES = {
    "workclass": {
        "Private": "Empresa privada", "Self-emp-not-inc": "Independiente (sin empresa constituida)",
        "Self-emp-inc": "Independiente (empresa constituida)", "Federal-gov": "Gobierno federal",
        "Local-gov": "Gobierno local", "State-gov": "Gobierno estatal",
        "Without-pay": "Sin remuneración", "Never-worked": "Nunca ha trabajado",
    },
    "education": {
        "Preschool": "Preescolar", "1st-4th": "Primaria 1.º-4.º", "5th-6th": "Primaria 5.º-6.º",
        "7th-8th": "Grados 7.º-8.º", "9th": "Grado 9.º", "10th": "Grado 10.º", "11th": "Grado 11.º",
        "12th": "Grado 12.º (sin diploma)", "HS-grad": "Bachiller (secundaria completa)",
        "Some-college": "Universidad incompleta", "Assoc-voc": "Técnico / vocacional",
        "Assoc-acdm": "Tecnólogo / asociado académico", "Bachelors": "Pregrado (profesional)",
        "Masters": "Maestría", "Prof-school": "Escuela profesional (medicina, derecho)",
        "Doctorate": "Doctorado",
    },
    "marital_status": {
        "Married-civ-spouse": "Casado/a (cónyuge civil)", "Married-AF-spouse": "Casado/a (cónyuge militar)",
        "Married-spouse-absent": "Casado/a (cónyuge ausente)", "Never-married": "Soltero/a",
        "Divorced": "Divorciado/a", "Separated": "Separado/a", "Widowed": "Viudo/a",
    },
    "occupation": {
        "Tech-support": "Soporte técnico", "Craft-repair": "Oficios y reparación",
        "Other-service": "Otros servicios", "Sales": "Ventas", "Exec-managerial": "Directivo / gerencial",
        "Prof-specialty": "Profesional especializado", "Handlers-cleaners": "Manipulación y limpieza",
        "Machine-op-inspct": "Operación e inspección de máquinas", "Adm-clerical": "Administrativo / oficina",
        "Farming-fishing": "Agricultura y pesca", "Transport-moving": "Transporte",
        "Priv-house-serv": "Servicio doméstico", "Protective-serv": "Servicios de protección",
        "Armed-Forces": "Fuerzas armadas",
    },
    "relationship": {
        "Husband": "Esposo", "Wife": "Esposa", "Own-child": "Hijo/a", "Not-in-family": "Sin familia en el hogar",
        "Other-relative": "Otro familiar", "Unmarried": "No casado/a (jefe de hogar)",
    },
    "race": {
        "White": "Blanca", "Black": "Negra", "Asian-Pac-Islander": "Asiática / isleña del Pacífico",
        "Amer-Indian-Eskimo": "Indígena americana / esquimal", "Other": "Otra",
    },
    "sex": {"Male": "Hombre", "Female": "Mujer"},
}

ETIQUETAS = {
    "age": "Edad (años)", "workclass": "Tipo de empleador", "education": "Nivel educativo",
    "marital_status": "Estado civil", "occupation": "Ocupación", "relationship": "Rol en el hogar",
    "race": "Raza (categoría del censo)", "sex": "Sexo", "capital_gain": "Ganancias de capital (USD/año)",
    "capital_loss": "Pérdidas de capital (USD/año)", "hours_per_week": "Horas trabajadas por semana",
    "native_country": "País de origen",
}

AYUDAS = {
    "age": "Edad en años cumplidos. El censo incluye personas de 17 a 90 años.",
    "workclass": "Quién paga el salario de la persona.",
    "education": "Máximo nivel educativo alcanzado.",
    "marital_status": "Estado civil según el censo de EE. UU. de 1994.",
    "occupation": "Tipo de trabajo que desempeña.",
    "relationship": "Relación de la persona con el jefe del hogar.",
    "race": "Categoría del censo. Se incluye porque está en los datos originales.",
    "sex": "Sexo registrado en el censo.",
    "capital_gain": "Dinero ganado en el año al vender inversiones o activos. El 92 % de las personas tiene 0.",
    "capital_loss": "Dinero perdido en el año al vender inversiones o activos. El 95 % de las personas tiene 0.",
    "hours_per_week": "Horas de trabajo habituales en una semana (40 = tiempo completo).",
    "native_country": "País de nacimiento, con los nombres originales del censo.",
}

LIMITES = {"age": (17, 90), "hours_per_week": (1, 99), "capital_gain": (0, 99999), "capital_loss": (0, 4356)}

# ============================================== arquetipos de persona para la demostración
ARQUETIPOS = {
    "andres": {
        "nombre": "Andrés, 45 años", "rol": "Gerente con maestría", "img": "persona_ejecutivo.png",
        "desc": "Casado, empresa privada, 50 h/semana y ganancias por inversiones.",
        "valores": {"age": 45, "workclass": "Private", "education": "Masters",
                    "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
                    "relationship": "Husband", "race": "White", "sex": "Male", "capital_gain": 15024,
                    "capital_loss": 0, "hours_per_week": 50, "native_country": "United-States"},
    },
    "laura": {
        "nombre": "Laura, 23 años", "rol": "Auxiliar de servicios", "img": "persona_servicios.png",
        "desc": "Bachiller, soltera, vive con su familia y trabaja medio tiempo (25 h).",
        "valores": {"age": 23, "workclass": "Private", "education": "HS-grad",
                    "marital_status": "Never-married", "occupation": "Other-service",
                    "relationship": "Own-child", "race": "White", "sex": "Female", "capital_gain": 0,
                    "capital_loss": 0, "hours_per_week": 25, "native_country": "United-States"},
    },
    "carolina": {
        "nombre": "Carolina, 30 años", "rol": "Técnica con maestría", "img": "persona_tecnica.png",
        "desc": "Casada, oficios y reparación, 40 h. Caso límite: los dos modelos no coinciden.",
        "valores": {"age": 30, "workclass": "Private", "education": "Masters",
                    "marital_status": "Married-civ-spouse", "occupation": "Craft-repair",
                    "relationship": "Wife", "race": "White", "sex": "Female", "capital_gain": 0,
                    "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States"},
    },
    "diego": {
        "nombre": "Diego, 38 años", "rol": "Docente con doctorado", "img": "persona_docente.png",
        "desc": "Casado, trabaja en el gobierno local como profesional, 45 h/semana.",
        "valores": {"age": 38, "workclass": "Local-gov", "education": "Doctorate",
                    "marital_status": "Married-civ-spouse", "occupation": "Prof-specialty",
                    "relationship": "Husband", "race": "Black", "sex": "Male", "capital_gain": 0,
                    "capital_loss": 0, "hours_per_week": 45, "native_country": "United-States"},
    },
}

DESCRIPCION_MODELOS = {
    "Random Forest": "300 árboles de decisión que votan. Es el más preciso: cuando dice «>50K» se equivoca menos.",
    "Perceptrón multicapa (MLP)": "Red neuronal de 2 capas ocultas (64 y 32 neuronas). Detecta un poco más de casos >50K.",
}


# ============================================================ utilidades de recursos
@st.cache_data
def b64(ruta):
    return base64.b64encode(Path(ruta).read_bytes()).decode()


def img_tag(nombre, clase="", alt="", estilo=""):
    ruta = DIR_IMG / nombre
    if not ruta.exists():
        return ""
    mime = "image/png" if nombre.endswith(".png") else "image/jpeg"
    return f'<img class="{clase}" alt="{alt}" style="{estilo}" src="data:{mime};base64,{b64(ruta)}">'


@st.cache_data
def css_fuentes():
    reglas = []
    for archivo, peso in (("Poppins-Regular.ttf", 400), ("Poppins-Medium.ttf", 500), ("Poppins-Bold.ttf", 700)):
        ruta = DIR_FUENTES / archivo
        if ruta.exists():
            reglas.append(f"@font-face{{font-family:'Poppins';font-weight:{peso};font-style:normal;"
                          f"src:url(data:font/ttf;base64,{b64(ruta)}) format('truetype');}}")
    return "\n".join(reglas)


def inyectar_estilos():
    st.markdown(f"""<style>
{css_fuentes()}
:root {{ --rojo:{ROJO}; --rojo-osc:{ROJO_OSCURO}; --carbon:{CARBON}; --teal:{TEAL}; --gris:{GRIS}; }}
html, body, .stApp, .stMarkdown, button, input, textarea, label, [data-testid="stSidebar"] *
  {{ font-family:'Poppins','Segoe UI',system-ui,sans-serif; }}
/* Los íconos de Streamlit (p. ej. el botón para plegar la barra lateral) usan una fuente de
   íconos: se restaura para que no aparezca su nombre en texto ("keyboard_double_arrow_left"). */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons,
[data-testid="stSidebar"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] span, [data-testid="stExpandSidebarButton"] span
  {{ font-family:'Material Symbols Rounded','Material Icons' !important; }}
.block-container {{ padding-top:1.4rem; max-width:1250px; }}
h1, h2, h3 {{ color:var(--carbon); letter-spacing:-.01em; }}

/* ---------- portada ---------- */
.hero {{ position:relative; border-radius:22px; overflow:hidden; min-height:230px; margin-bottom:1.2rem;
        background-size:cover; background-position:center 60%; box-shadow:0 10px 30px rgba(0,0,0,.18); }}
.hero::before {{ content:""; position:absolute; inset:0;
        background:linear-gradient(100deg, rgba(163,0,13,.94) 0%, rgba(227,6,19,.80) 42%, rgba(29,29,31,.15) 100%); }}
.hero-in {{ position:relative; padding:28px 34px; color:#fff; max-width:760px; }}
.hero-in img.logo {{ height:40px; margin-bottom:14px; }}
.hero-in h1 {{ color:#fff; font-size:2.25rem; font-weight:700; line-height:1.15; margin:0 0 .4rem 0; padding:0; }}
.hero-in p {{ font-size:1.02rem; opacity:.96; margin:0 0 .9rem 0; }}

/* ---------- ficha del proyecto (debajo de la portada) ---------- */
.ficha {{ display:grid; grid-template-columns:repeat(4,1fr); background:#fff; border:1px solid #ECECEF; border-radius:16px;
         margin:-.4rem 0 1.1rem 0; overflow:hidden; }}
.ficha div {{ padding:12px 16px; border-right:1px solid #ECECEF; }}
.ficha div:last-child {{ border-right:0; }}
.ficha .et {{ display:block; font-size:.7rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--rojo); margin-bottom:2px; }}
.ficha .va {{ font-size:.86rem; color:#374151; line-height:1.35; }}
.sb-intro {{ background:#FDECEE; border-radius:14px; padding:12px 14px; font-size:.84rem; color:#374151; margin-bottom:.6rem; }}
.sb-intro b {{ color:var(--carbon); }}

/* ---------- pasos ---------- */
.pasos {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:.2rem 0 1.4rem 0; }}
.paso {{ background:#fff; border:1px solid #ECECEF; border-radius:16px; padding:14px 16px; display:flex; gap:12px; align-items:flex-start; }}
.paso .n {{ flex:0 0 34px; height:34px; border-radius:50%; background:var(--rojo); color:#fff; font-weight:700;
           display:flex; align-items:center; justify-content:center; }}
.paso b {{ color:var(--carbon); }} .paso span {{ color:var(--gris); font-size:.86rem; }}

/* ---------- títulos de sección ---------- */
.sec {{ display:flex; align-items:center; gap:10px; margin:1.1rem 0 .2rem 0; }}
.sec .n {{ background:var(--carbon); color:#fff; border-radius:8px; padding:2px 10px; font-weight:700; font-size:.9rem; }}
.sec h3 {{ margin:0; padding:0; font-size:1.25rem; }}
.sub {{ color:var(--gris); font-size:.9rem; margin:0 0 .8rem 0; }}

/* ---------- tarjetas de arquetipo ---------- */
.card {{ background:#fff; border:1px solid #ECECEF; border-radius:18px; padding:16px 14px 10px; text-align:center;
        transition:all .15s ease; min-height:250px; }}
.card:hover {{ border-color:var(--rojo); box-shadow:0 8px 22px rgba(227,6,19,.12); transform:translateY(-2px); }}
.card.activa {{ border:2px solid var(--rojo); box-shadow:0 8px 22px rgba(227,6,19,.16); }}
.card img {{ width:96px; height:96px; }}
.card .nom {{ font-weight:700; color:var(--carbon); margin-top:6px; font-size:.98rem; }}
.card .rol {{ color:var(--rojo); font-weight:500; font-size:.84rem; }}
.card .des {{ color:var(--gris); font-size:.78rem; line-height:1.35; margin-top:6px; }}
.stButton > button {{ width:100%; border-radius:12px; font-weight:500; }}

/* ---------- grupos del formulario ---------- */
.grupo {{ font-weight:700; color:var(--carbon); font-size:1rem; margin:.1rem 0 .4rem 0; }}
.grupo small {{ display:block; color:var(--gris); font-weight:400; font-size:.78rem; }}

/* ---------- botón predecir ---------- */
div[data-testid="stButton"] button[kind="primary"] {{ font-size:1.1rem; font-weight:700; padding:.7rem 0;
        background:var(--rojo); border-color:var(--rojo); box-shadow:0 8px 18px rgba(227,6,19,.28); }}
div[data-testid="stButton"] button[kind="primary"]:hover {{ background:var(--rojo-osc); border-color:var(--rojo-osc); }}

/* ---------- resultado ---------- */
.res {{ border-radius:20px; padding:22px 26px; display:flex; gap:24px; align-items:center; margin-top:.6rem;
       border:1px solid #ECECEF; background:#fff; box-shadow:0 10px 28px rgba(0,0,0,.08); }}
.res.alto {{ border-left:0; background:linear-gradient(90deg,#E8F5F2 0%,#FFFFFF 55%); }}
.res.bajo {{ background:linear-gradient(90deg,#F1F2F4 0%,#FFFFFF 55%); }}
.res img {{ width:130px; height:130px; flex:0 0 130px; }}
.res .tag {{ font-size:.8rem; font-weight:500; color:var(--gris); text-transform:uppercase; letter-spacing:.06em; }}
.res .clase {{ font-size:1.7rem; font-weight:700; color:var(--carbon); line-height:1.2; margin:.15rem 0 .3rem; }}
.res .clase em {{ font-style:normal; color:var(--teal); }} .res.bajo .clase em {{ color:#4B5563; }}
.gauge {{ position:relative; height:16px; border-radius:999px; background:#E9E9EC; margin:12px 0 6px; }}
.gauge .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:999px;
               background:linear-gradient(90deg,#F59AA0,var(--rojo)); }}
.gauge .mark {{ position:absolute; left:50%; top:-6px; bottom:-6px; width:2px; background:var(--carbon); }}
.gauge-lbl {{ display:flex; justify-content:space-between; font-size:.75rem; color:var(--gris); }}
.big {{ font-size:2.3rem; font-weight:700; color:var(--rojo); line-height:1; }}
.nota {{ font-size:.85rem; color:#374151; margin-top:.5rem; }}

/* ---------- barra lateral ---------- */
[data-testid="stSidebar"] {{ background:#FAFAFB; border-right:1px solid #ECECEF; }}
.sb-card {{ background:#fff; border:1px solid #ECECEF; border-radius:14px; padding:12px 14px; font-size:.82rem; color:#374151; }}
.sb-card b {{ color:var(--carbon); }}
.pie {{ text-align:center; color:var(--gris); font-size:.78rem; margin-top:2rem; padding-top:1rem; border-top:1px solid #ECECEF; }}
@media (max-width: 900px) {{ .ficha {{ grid-template-columns:1fr 1fr; }} .pasos {{ grid-template-columns:1fr; }} .res {{ flex-direction:column; text-align:center; }} }}
</style>""", unsafe_allow_html=True)


# ====================================================================== carga de datos
@st.cache_data
def cargar_metadatos(ruta=RUTA_METADATOS):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo(nombre_archivo):
    """Carga un Pipeline completo (preprocesamiento + modelo) una sola vez por sesión del servidor."""
    return joblib.load(DIR_MODELOS / nombre_archivo)


# ================================================================= lógica de inferencia
def construir_registro(valores, features):
    """DataFrame de una fila con las columnas en el MISMO orden usado en el entrenamiento."""
    faltan = [c for c in features if c not in valores]
    if faltan:
        raise ValueError(f"Faltan campos obligatorios: {', '.join(faltan)}")
    return pd.DataFrame([[valores[c] for c in features]], columns=features)


def validar(valores, meta):
    """Devuelve (errores, advertencias). Los errores impiden predecir; las advertencias solo informan."""
    errores, advertencias = [], []
    for col in meta["numericas"]:
        v = valores.get(col)
        lo, hi = LIMITES.get(col, (None, None))
        if v is None or pd.isna(v):
            errores.append(f"«{ETIQUETAS[col]}» es obligatorio.")
        elif lo is not None and not (lo <= v <= hi):
            errores.append(f"«{ETIQUETAS[col]}» debe estar entre {lo:,} y {hi:,}.")
    for col in meta["categoricas"]:
        if valores.get(col) not in meta["categorias"][col]:
            errores.append(f"«{ETIQUETAS[col]}» tiene un valor no válido.")
    if errores:
        return errores, advertencias

    if valores["capital_gain"] > 0 and valores["capital_loss"] > 0:
        advertencias.append("Es inusual reportar a la vez ganancias y pérdidas de capital; en los datos casi nunca ocurre.")
    if valores["relationship"] == "Husband" and valores["sex"] == "Female":
        advertencias.append("Rol en el hogar «Esposo» con sexo «Mujer» es una combinación atípica en los datos.")
    if valores["relationship"] == "Wife" and valores["sex"] == "Male":
        advertencias.append("Rol en el hogar «Esposa» con sexo «Hombre» es una combinación atípica en los datos.")
    if valores["relationship"] in ("Husband", "Wife") and not valores["marital_status"].startswith("Married"):
        advertencias.append("El rol «Esposo/Esposa» no es coherente con un estado civil distinto de casado/a.")
    if valores["workclass"] == "Never-worked" and valores["hours_per_week"] > 0:
        advertencias.append("«Nunca ha trabajado» con horas semanales mayores que 0 es una combinación atípica.")
    if valores["age"] < 25 and valores["education"] in ("Doctorate", "Prof-school"):
        advertencias.append("Edad muy baja para el nivel educativo seleccionado.")
    return errores, advertencias


def predecir(modelo, registro):
    """Devuelve (clase 0/1, probabilidad de >50K o None)."""
    clase = int(modelo.predict(registro)[0])
    prob = float(modelo.predict_proba(registro)[0, 1]) if hasattr(modelo, "predict_proba") else None
    return clase, prob


def etiqueta(col, valor):
    trad = TRADUCCIONES.get(col, {}).get(valor)
    return f"{trad} ({valor})" if trad else valor


def nivel_confianza(prob):
    c = max(prob, 1 - prob)
    if c >= 0.85:
        return "alta", "El modelo está muy seguro de esta clasificación."
    if c >= 0.65:
        return "media", "El modelo se inclina con claridad, pero no es una certeza."
    return "baja", "Caso dudoso: la probabilidad está cerca del umbral de 50 %. Prueba con el otro modelo."


# ======================================================================== estado
def valores_por_defecto(meta):
    v = {c: meta["moda"][c] for c in meta["categoricas"]}
    for c in meta["numericas"]:
        lo, hi = LIMITES[c]
        v[c] = int(min(max(meta["rangos"][c]["mediana"], lo), hi))
    return v


def cargar_perfil(clave, meta):
    valores = valores_por_defecto(meta) if clave == "personalizado" else ARQUETIPOS[clave]["valores"]
    for k, v in valores.items():
        st.session_state[k] = v
    st.session_state["perfil"] = clave
    st.session_state.pop("resultado", None)


def limpiar_resultado():
    st.session_state.pop("resultado", None)


# ===================================================================== componentes
def portada():
    fondo = DIR_IMG / "campus_atardecer.jpg"
    estilo = f"background-image:url(data:image/jpeg;base64,{b64(fondo)});" if fondo.exists() else f"background:{ROJO};"
    st.markdown(f"""<div class="hero" style="{estilo}"><div class="hero-in">
{img_tag("logo_uao_horizontal_blanco.png", "logo", "Universidad Autónoma de Occidente")}
<h1>Predictor de nivel de ingreso</h1>
<p>Estima si una persona gana <b>más de 50 000 USD al año</b> a partir de su perfil demográfico y laboral,
con dos modelos de Machine Learning entrenados en Google Colab.</p>
</div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="ficha">
<div><span class="et">¿Qué predice?</span><span class="va">Si el ingreso anual es mayor o menor a 50 000 USD (clasificación binaria)</span></div>
<div><span class="et">¿Con qué datos?</span><span class="va">Censo de EE. UU. de 1994: dataset Adult / Census Income de UCI</span></div>
<div><span class="et">¿Con qué modelos?</span><span class="va">Random Forest y Perceptrón multicapa (MLP)</span></div>
<div><span class="et">¿Qué hace esta app?</span><span class="va">Solo predice: los modelos ya se entrenaron en Colab y aquí no se reentrenan</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="pasos">
<div class="paso"><div class="n">1</div><div><b>Elige el modelo</b><br><span>En la barra lateral izquierda.</span></div></div>
<div class="paso"><div class="n">2</div><div><b>Elige un perfil</b><br><span>Carga un arquetipo de persona.</span></div></div>
<div class="paso"><div class="n">3</div><div><b>Ajusta los datos</b><br><span>Cambia cualquier característica si quieres.</span></div></div>
<div class="paso"><div class="n">4</div><div><b>Pulsa «Predecir»</b><br><span>Verás la clase y la probabilidad de ganar más de 50K.</span></div></div>
</div>""", unsafe_allow_html=True)


def titulo_seccion(numero, texto, subtitulo):
    st.markdown(f'<div class="sec"><span class="n">{numero}</span><h3>{texto}</h3></div>'
                f'<p class="sub">{subtitulo}</p>', unsafe_allow_html=True)


def barra_lateral(meta):
    with st.sidebar:
        st.markdown(f'<div style="text-align:center;margin:-.5rem 0 .6rem">{img_tag("logo_uao_escudo.png", "", "UAO", "width:120px")}</div>',
                    unsafe_allow_html=True)
        st.markdown("### ⚙️ Paso 1 · Elige el modelo")
        st.markdown("""<div class="sb-intro">Aquí decides qué modelo hará la predicción. Los dos aprendieron
de los mismos datos del censo, pero de forma distinta. Si no sabes cuál usar, empieza con Random Forest,
el de mejor desempeño. Después de predecir puedes cambiar de modelo y pulsar «Predecir» otra vez para comparar.</div>""",
                    unsafe_allow_html=True)
        nombres = list(meta["modelos"].keys())
        elegidas = meta.get("variaciones_elegidas", {})
        nombre = st.radio("Algoritmo entrenado", nombres, key="modelo", on_change=limpiar_resultado,
                          captions=[DESCRIPCION_MODELOS.get(n, "") for n in nombres],
                          help="Ambos modelos se entrenaron con los mismos datos y el mismo preprocesamiento. "
                               "Cambia de modelo y vuelve a pulsar «Predecir» para comparar.")
        if nombre in elegidas:
            st.caption(f"Variación elegida en Colab: **{elegidas[nombre]}**")
        st.caption(f"Archivo cargado: `modelos/{meta['modelos'][nombre]}`")

        st.markdown("### 📊 Datos")
        st.markdown(f"""<div class="sb-card"><b>Adult / Census Income</b><br>
Censo de EE. UU. de 1994 · 48 842 personas · 12 variables de entrada.<br>
<a href="{meta['url_dataset']}" target="_blank">UCI Machine Learning Repository ↗</a><br>
<span style="color:{GRIS}">Becker & Kohavi (1996), CC BY 4.0 · doi:10.24432/C5XW20</span></div>""",
                    unsafe_allow_html=True)

        st.markdown("### ❓ ¿Cómo leer el resultado?")
        st.markdown("""<div class="sb-card">La app muestra la clase predicha (&gt;50K o ≤50K) y la
probabilidad que el modelo asigna a &gt;50K. Si la probabilidad es de 50 % o más, la clase es &gt;50K.
Cuanto más lejos de 50 %, más seguro está el modelo.</div>""", unsafe_allow_html=True)
        return nombre


def selector_arquetipos(meta):
    titulo_seccion("2", "Elige un arquetipo de persona",
                   "Cada tarjeta carga un perfil completo en el formulario. Luego puedes modificar lo que quieras.")
    actual = st.session_state.get("perfil")
    claves = list(ARQUETIPOS) + ["personalizado"]
    cols = st.columns(len(claves))
    for col, clave in zip(cols, claves):
        if clave == "personalizado":
            a = {"nombre": "Perfil propio", "rol": "Desde cero", "img": "persona_personalizado.png",
                 "desc": "Carga valores típicos del censo (moda y mediana) y edítalos a tu gusto."}
        else:
            a = ARQUETIPOS[clave]
        with col:
            st.markdown(f"""<div class="card {'activa' if actual == clave else ''}">
{img_tag(a['img'], '', a['nombre'])}
<div class="nom">{a['nombre']}</div><div class="rol">{a['rol']}</div><div class="des">{a['desc']}</div></div>""",
                        unsafe_allow_html=True)
            st.button("✓ Perfil cargado" if actual == clave else "Usar este perfil", key=f"btn_{clave}",
                      on_click=cargar_perfil, args=(clave, meta), type="secondary")


def formulario(meta):
    titulo_seccion("3", "Revisa o ajusta las características",
                   "Pasa el cursor sobre el ícono ⓘ de cada campo para ver qué significa. "
                   "Los valores se envían al modelo tal como los usó el censo.")
    cat = meta["categorias"]
    with st.container(border=True):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown('<div class="grupo">👤 Perfil personal<small>Datos demográficos</small></div>', unsafe_allow_html=True)
            st.number_input(ETIQUETAS["age"], *LIMITES["age"], step=1, key="age", help=AYUDAS["age"],
                            on_change=limpiar_resultado)
            st.selectbox(ETIQUETAS["sex"], cat["sex"], format_func=lambda v: etiqueta("sex", v), key="sex",
                         help=AYUDAS["sex"], on_change=limpiar_resultado)
            st.selectbox(ETIQUETAS["race"], cat["race"], format_func=lambda v: etiqueta("race", v), key="race",
                         help=AYUDAS["race"], on_change=limpiar_resultado)
            st.selectbox(ETIQUETAS["native_country"], cat["native_country"], key="native_country",
                         help=AYUDAS["native_country"], on_change=limpiar_resultado)
        with c2:
            st.markdown('<div class="grupo">🏠 Hogar y educación<small>Familia y formación</small></div>', unsafe_allow_html=True)
            st.selectbox(ETIQUETAS["marital_status"], cat["marital_status"],
                         format_func=lambda v: etiqueta("marital_status", v), key="marital_status",
                         help=AYUDAS["marital_status"], on_change=limpiar_resultado)
            st.selectbox(ETIQUETAS["relationship"], cat["relationship"],
                         format_func=lambda v: etiqueta("relationship", v), key="relationship",
                         help=AYUDAS["relationship"], on_change=limpiar_resultado)
            st.selectbox(ETIQUETAS["education"], cat["education"], format_func=lambda v: etiqueta("education", v),
                         key="education", help=AYUDAS["education"], on_change=limpiar_resultado)
        with c3:
            st.markdown('<div class="grupo">💼 Trabajo y finanzas<small>Empleo, jornada y capital</small></div>', unsafe_allow_html=True)
            st.selectbox(ETIQUETAS["workclass"], cat["workclass"], format_func=lambda v: etiqueta("workclass", v),
                         key="workclass", help=AYUDAS["workclass"], on_change=limpiar_resultado)
            st.selectbox(ETIQUETAS["occupation"], cat["occupation"], format_func=lambda v: etiqueta("occupation", v),
                         key="occupation", help=AYUDAS["occupation"], on_change=limpiar_resultado)
            st.slider(ETIQUETAS["hours_per_week"], *LIMITES["hours_per_week"], key="hours_per_week",
                      help=AYUDAS["hours_per_week"], on_change=limpiar_resultado)
            g1, g2 = st.columns(2)
            with g1:
                st.number_input(ETIQUETAS["capital_gain"], *LIMITES["capital_gain"], step=500, key="capital_gain",
                                help=AYUDAS["capital_gain"], on_change=limpiar_resultado)
            with g2:
                st.number_input(ETIQUETAS["capital_loss"], *LIMITES["capital_loss"], step=100, key="capital_loss",
                                help=AYUDAS["capital_loss"], on_change=limpiar_resultado)
    return {c: st.session_state.get(c) for c in meta["features"]}


def mostrar_resultado(r):
    clase, prob, nombre_modelo = r["clase"], r["prob"], r["modelo"]
    alto = clase == 1
    nivel, texto_nivel = nivel_confianza(prob)
    perfil = r.get("perfil_nombre") or "la persona ingresada"
    st.markdown(f"""<div class="res {'alto' if alto else 'bajo'}">
{img_tag('resultado_alto.png' if alto else 'resultado_bajo.png', '', 'resultado')}
<div style="flex:1">
  <div class="tag">Resultado · {nombre_modelo}</div>
  <div class="clase">Para {perfil}, el modelo estima un ingreso <em>{'mayor a 50 000 USD/año' if alto else 'menor o igual a 50 000 USD/año'}</em></div>
  <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
     <span class="big">{prob:.1%}</span><span style="color:{GRIS}">de probabilidad de ganar más de 50K</span></div>
  <div class="gauge"><div class="fill" style="width:{prob*100:.1f}%"></div><div class="mark"></div></div>
  <div class="gauge-lbl"><span>0 %</span><span>umbral 50 %</span><span>100 %</span></div>
  <div class="nota">Confianza <b>{nivel}</b> ({max(prob, 1-prob):.0%}). {texto_nivel}</div>
</div></div>""", unsafe_allow_html=True)
    with st.expander("Ver los datos exactos que recibió el modelo (mismo orden de columnas que en el entrenamiento)"):
        st.dataframe(r["registro"], hide_index=True)
    st.caption("La probabilidad es la estimación del modelo, no una certeza. El modelo aprendió patrones del censo "
               "de EE. UU. de 1994, incluidas sus desigualdades; no debe usarse para decisiones sobre personas reales.")


# ========================================================================= principal
def main():
    st.set_page_config(page_title="Predictor de ingreso · UAO", page_icon="💼", layout="wide",
                       initial_sidebar_state="expanded")
    inyectar_estilos()

    try:
        meta = cargar_metadatos()
    except FileNotFoundError:
        st.error(f"No se encontró {RUTA_METADATOS}. Ejecute el notebook (Celda 12) y copie la carpeta "
                 "«modelos» junto a app.py.")
        st.stop()
    except json.JSONDecodeError:
        st.error("El archivo modelos/metadatos.json está dañado. Vuelva a generarlo con el notebook.")
        st.stop()
    faltantes = [a for a in meta["modelos"].values() if not (DIR_MODELOS / a).exists()]
    if faltantes:
        st.error("Faltan archivos de modelo en la carpeta «modelos»: " + ", ".join(faltantes))
        st.stop()

    for k, v in valores_por_defecto(meta).items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("perfil", "personalizado")

    nombre_modelo = barra_lateral(meta)
    portada()
    selector_arquetipos(meta)
    valores = formulario(meta)

    # ¿El formulario sigue igual al arquetipo cargado?
    perfil = st.session_state.get("perfil")
    perfil_nombre = None
    if perfil in ARQUETIPOS:
        perfil_nombre = ARQUETIPOS[perfil]["nombre"].split(",")[0]
        if any(valores[k] != v for k, v in ARQUETIPOS[perfil]["valores"].items()):
            perfil_nombre += " (modificado)"

    titulo_seccion("4", "Obtén la predicción", f"Modelo seleccionado: <b>{nombre_modelo}</b>. "
                   "Puedes cambiarlo en la barra lateral y volver a predecir.")
    _, centro, _ = st.columns([1, 2, 1])
    with centro:
        pulsado = st.button("🔮  Predecir", type="primary", key="predecir")

    if pulsado:
        errores, advertencias = validar(valores, meta)
        if errores:
            for e in errores:
                st.error(e)
            st.stop()
        for a in advertencias:
            st.warning(a)
        try:
            registro = construir_registro(valores, meta["features"])
            modelo = cargar_modelo(meta["modelos"][nombre_modelo])
            with st.spinner("Calculando…"):
                clase, prob = predecir(modelo, registro)
        except Exception as e:  # versiones de librerías incompatibles, archivo dañado, etc.
            st.error("No fue posible generar la predicción. Verifique que las versiones de scikit-learn, numpy y "
                     "pandas coincidan con requirements.txt y que los archivos de «modelos» no estén dañados.\n\n"
                     f"Detalle técnico: {type(e).__name__}: {e}")
            st.stop()
        st.session_state["resultado"] = {"clase": clase, "prob": prob, "modelo": nombre_modelo,
                                         "registro": registro, "perfil_nombre": perfil_nombre}

    if "resultado" in st.session_state:
        mostrar_resultado(st.session_state["resultado"])

    st.markdown(f"""<div class="pie">{img_tag("logo_uao_corto.png", "", "uao", "height:22px;vertical-align:middle")}<br>
Maestría en Inteligencia Artificial y Ciencia de Datos · Aprendizaje Automático · Parcial práctico 1</div>""",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
