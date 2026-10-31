# Parcial práctico 1 — Predicción del nivel de ingreso (Adult / Census Income, UCI)

Clasificación binaria: ¿la persona gana **más de 50 000 USD al año**? Dos modelos (Random Forest y Perceptrón multicapa), cada uno con **dos variaciones** comparadas por validación cruzada, entrenados en Google Colab y servidos con Streamlit solo para inferencia.

**Integrantes:** Angie Tatiana Rodríguez Duque · Kevin Fernando Rodríguez Duque

- **Dataset (opción 7 del catálogo):** https://archive.ics.uci.edu/dataset/2/adult
- **Descarga directa:** https://archive.ics.uci.edu/static/public/2/adult.zip
- Cita: Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20 (CC BY 4.0)

---

## 1. Por qué este dataset

| Criterio del taller | Adult / Census Income |
|---|---|
| Facilidad de descarga | ZIP oficial de UCI de 620 KB; el notebook lo descarga solo. |
| Variable objetivo clara | `income`: `>50K` / `<=50K` (binaria, sin tener que inventar categorías como en Wine Quality). |
| Interpretación | Variables cotidianas (edad, educación, ocupación, horas), fáciles de explicar. |
| Compatibilidad con los algoritmos | Tabular, 48 842 filas: ideal para Random Forest y MLP. |
| Streamlit | 12 entradas: 4 numéricas (number_input / slider) y 8 categóricas (selectbox). |
| Recursos | Ambos modelos entrenan en segundos en Colab. |
| Criterio de 15 % (preparación) | Tiene **faltantes reales** (`?`), **duplicados**, **inconsistencias de etiqueta** (`>50K.`) y **8 variables categóricas** que codificar: permite mostrar todo el preprocesamiento exigido. |

## 2. Definición del problema

- **Aprendizaje supervisado — clasificación binaria.**
- **y:** `income` → 1 si `>50K`, 0 si `<=50K` (≈ 24 % positivos: clases desbalanceadas).
- **X (12):** `age`, `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `capital_gain`, `capital_loss`, `hours_per_week`, `native_country`.
- **Excluidas:** `fnlwgt` (peso muestral del censo, no describe a la persona; correlación ≈ 0) y `education_num` (recodificación 1 a 1 de `education`).
- **Sin fuga de información:** ninguna entrada es consecuencia del ingreso; `income` no está en X (se verifica con `assert`); el preprocesamiento se ajusta solo con entrenamiento (Pipeline); se eliminan duplicados antes de dividir.

## 3. Estructura del proyecto (contenido del ZIP)

```
Predictor_Nivel_Ingreso/
├── notebook_parcial1.ipynb          # Notebook de Colab (14 celdas, ya ejecutado con salidas)
├── app.py                           # Aplicación Streamlit (solo inferencia)
├── .streamlit/config.toml           # Tema de colores UAO para Streamlit (carpeta oculta: no la borre)
├── assets/
│   ├── img/                         # Logos y fotos UAO, ilustraciones de arquetipos y de resultado
│   └── fonts/                       # Tipografía Poppins (la app funciona sin internet)
├── requirements.txt                 # Dependencias con versiones exactas
├── README.md                        # Este archivo
├── GUIA_SUSTENTACION.md             # Guía para defender el proyecto + 10 preguntas + checklist
├── datos/
│   ├── adult.zip                    # ZIP oficial descargado de UCI
│   ├── adult.data                   # 32 561 registros (partición original de entrenamiento)
│   ├── adult.test                   # 16 281 registros (partición original de prueba)
│   ├── adult.names                  # Descripción oficial de las variables
│   └── Index, old.adult.names       # Archivos que vienen dentro del ZIP oficial
└── modelos/
    ├── modelo_random_forest.joblib  # Pipeline completo: preprocesamiento + Random Forest
    ├── modelo_mlp.joblib            # Pipeline completo: preprocesamiento + MLP
    └── metadatos.json               # Orden de columnas, categorías válidas, rangos, métricas, versiones
```

**El preprocesamiento está dentro de cada `.joblib`** (imputación, `log1p`, escalado y one-hot ajustados con entrenamiento). No hay un archivo de preprocesamiento aparte: `joblib.load("modelos/modelo_mlp.joblib").predict(df)` recibe un DataFrame con las 12 columnas originales, en el orden de `metadatos.json["features"]`, y aplica todo internamente. `metadatos.json` solo lo usa la app para construir y validar los campos.

Las rutas son relativas: el notebook crea `datos/` y `modelos/` en su carpeta de trabajo y `app.py` busca `modelos/` junto a sí mismo (`Path(__file__).parent / "modelos"`), así que funciona desde cualquier ubicación siempre que se conserve esta estructura.

### Variaciones entrenadas (Celdas 8 y 9)

| Variación | Hiperparámetros | F1 val (CV 5) | Brecha train−val | Archivo | Decisión |
|---|---|---|---|---|---|
| RF-A · regularizado | 300 árboles, `max_depth=20`, `min_samples_leaf=5` | **0,664** | **0,017** | 8,2 MB | ✅ elegida |
| RF-B · árboles completos | 300 árboles, `max_depth=None`, `min_samples_leaf=1` | 0,655 | 0,295 | 53,2 MB | descartada |
| MLP-A · compacta + early stopping | capas (64, 32), `alpha=1e-3`, early stopping | **0,666** | **0,035** | 205 KB | ✅ elegida |
| MLP-B · amplia sin early stopping | capas (128, 64), `alpha=1e-4`, sin early stopping | 0,624 | 0,225 | 657 KB | descartada |

La selección se hizo solo con el 80 % de entrenamiento (validación cruzada estratificada de 5 pliegues); el 20 % de prueba se reserva para comparar los dos modelos finales. La re-ejecución completa del notebook tarda ≈ 6–8 minutos en Colab por la validación cruzada.

### La aplicación

Diseño con la identidad de la UAO (rojo institucional, logos, foto del campus, tipografía Poppins incluida). Debajo de la portada, una ficha explica qué predice la app, con qué datos y con qué modelos. Flujo guiado en 4 pasos: (1) elegir el modelo en la barra lateral, que incluye una breve introducción → (2) cargar un **arquetipo de persona** (Andrés, Laura, Carolina, Diego o perfil propio) → (3) ajustar los 12 campos → (4) **Predecir**. El resultado muestra la clase, la probabilidad, una barra con el umbral de 50 % y el nivel de confianza.

## 4. requirements.txt

```
# Versiones registradas al entrenar (Celda 12 del notebook)
streamlit>=1.36
scikit-learn==1.9.1
pandas==3.0.5
numpy==2.5.3
joblib==1.6.0
# Solo para ejecutar el notebook fuera de Colab:
matplotlib==3.11.2
seaborn==0.13.2
```

`scikit-learn`, `numpy`, `pandas` y `joblib` van **fijados** porque un modelo guardado con joblib debe cargarse con la misma versión de scikit-learn con que se entrenó. La Celda 12 del notebook **reescribe este archivo con las versiones del entorno donde se ejecute**: si vuelven a entrenar en Colab, usen el `requirements.txt` que genere Colab (y los modelos de esa misma ejecución).

Comprobar versiones instaladas: `python -m pip show scikit-learn numpy pandas joblib streamlit` o `python -m pip freeze`.

## 5. Ejecución

### A. Google Colab (entrenamiento)
1. Ir a https://colab.research.google.com → *Archivo → Subir cuaderno* → `notebook_parcial1.ipynb`.
2. *Entorno de ejecución → Ejecutar todas* (o ejecutar las celdas en orden con Shift+Enter).
   - La Celda 3 descarga el ZIP de UCI a `datos/` y verifica 48 842 × 15. Si falla, descargue el ZIP de la ficha de UCI y súbalo a `datos/` con el panel de archivos (ícono de carpeta) y vuelva a ejecutar la celda.
3. Las celdas 8–11 entrenan, evalúan y comparan los modelos (≈ 10 s en total).
4. La Celda 12 guarda `modelos/*.joblib`, `modelos/metadatos.json` y `requirements.txt` y comprueba que se cargan.
5. La Celda 14 crea y descarga `artefactos_parcial1.zip` (modelos + datos + requirements). Descomprímalo dentro de la carpeta del proyecto, reemplazando las carpetas `modelos/` y `datos/` y el `requirements.txt`.
6. Descargue el notebook ejecutado: *Archivo → Descargar → Descargar .ipynb* y guárdelo como `notebook_parcial1.ipynb`.

> El ZIP entregado ya trae modelos entrenados y el notebook ejecutado (scikit-learn 1.9.1). Si no re-entrenan en Colab, pueden pasar directamente a la parte B.

### B. En el computador (Streamlit)
1. Instalar Anaconda: https://www.anaconda.com/download
2. Abrir **Anaconda Prompt** y crear el entorno del taller:
   ```
   conda create -n parcial1 python=3.12 -y
   conda activate parcial1
   ```
3. Ir a la carpeta del proyecto (las comillas son necesarias por los espacios/tildes):
   ```
   cd "C:\ruta\a\Predictor_Nivel_Ingreso"
   ```
4. Instalar dependencias:
   ```
   python -m pip install -r requirements.txt
   ```
5. Ejecutar la app:
   ```
   streamlit run app.py
   ```
6. Abrir la dirección que muestra la terminal (normalmente http://localhost:8501). Para detener: Ctrl + C en la terminal.

### Errores frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `'streamlit' no se reconoce como un comando` | Entorno no activado | `conda activate parcial1` o usar `python -m streamlit run app.py`. |
| `ModuleNotFoundError: No module named 'sklearn'` (u otro) | Faltan dependencias en el entorno | `python -m pip install -r requirements.txt` con el entorno activado. |
| La app dice «No se encontró ... metadatos.json» o «Faltan archivos de modelo» | Carpeta `modelos/` no está junto a `app.py` | Copiar `modelos/` completa al lado de `app.py`; no renombrar los archivos. |
| `InconsistentVersionWarning` o error al cargar `.joblib` (p. ej. `AttributeError`, `ModuleNotFoundError: numpy._core`) | Versión de scikit-learn/numpy distinta a la del entrenamiento | Instalar exactamente las versiones de `requirements.txt` (`pip install scikit-learn==X.Y.Z`), o re-entrenar en Colab y usar el `requirements.txt` y los modelos de esa ejecución. |
| `ERROR: No matching distribution found for scikit-learn==...` | Esa versión no existe para su versión de Python | Verificar `python --version` (debe ser 3.12 con el entorno del taller); si persiste, re-ejecutar el notebook en Colab y usar el `requirements.txt` que genera. |
| `FileNotFoundError: datos/adult.data` en el notebook | Descarga fallida | Subir manualmente el ZIP de UCI a `datos/` y re-ejecutar la Celda 3. |
| Puerto 8501 ocupado | Otra app abierta | `streamlit run app.py --server.port 8502`. |

## 6. Entrega en ZIP

1. Verifique que la carpeta tenga exactamente la estructura de la sección 3 (notebook, `app.py`, `requirements.txt`, `.streamlit/`, `assets/`, `datos/`, `modelos/`).
2. Pruebe una última vez `streamlit run app.py` desde esa carpeta.
4. Windows: clic derecho sobre la carpeta `Predictor_Nivel_Ingreso` → *Enviar a → Carpeta comprimida (en zip)* (o *Comprimir en archivo ZIP* en Windows 11). Nombre sugerido: `Parcial1_ApellidoNombre1_ApellidoNombre2.zip`.
5. Abra el ZIP para confirmar que contiene `modelos/*.joblib` (el de Random Forest pesa ≈ 8 MB) y súbalo al campus virtual.
