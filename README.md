# Predicción del nivel de ingreso (Adult / Census Income, UCI)

Clasificación binaria: ¿la persona gana **más de 50 000 USD al año**? Dos modelos (Random Forest y Perceptrón multicapa), cada uno con dos variaciones comparadas por validación cruzada, entrenados en Google Colab y servidos con Streamlit solo para inferencia.

**Integrantes:** Angie Tatiana Rodríguez Duque · Kevin Fernando Rodríguez Duque

- **Dataset:** https://archive.ics.uci.edu/dataset/2/adult
- **Descarga directa:** https://archive.ics.uci.edu/static/public/2/adult.zip
- Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20 (CC BY 4.0)

---

## 1. Definición del problema

- **Aprendizaje supervisado — clasificación binaria.**
- **y:** `income` → 1 si `>50K`, 0 si `<=50K` (≈ 24 % positivos: clases desbalanceadas).
- **X (12):** `age`, `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `capital_gain`, `capital_loss`, `hours_per_week`, `native_country`.
- **Excluidas:** `fnlwgt` (peso muestral del censo, no describe a la persona; correlación ≈ 0) y `education_num` (recodificación 1 a 1 de `education`).
- **Sin fuga de información:** ninguna entrada es consecuencia del ingreso; `income` no está en X (se verifica con `assert`); el preprocesamiento se ajusta solo con entrenamiento (Pipeline); se eliminan duplicados antes de dividir.

## 2. Estructura del proyecto

```
Predictor_Nivel_Ingreso/
├── notebook_parcial1.ipynb          # Notebook de Colab
├── app.py                           # Aplicación Streamlit
├── .streamlit/config.toml           # Tema de colores para Streamlit
├── assets/
│   ├── img/                         # Logos y fotos, ilustraciones de arquetipos y de resultado
│   └── fonts/                       # Tipografía Poppins
├── requirements.txt                 # Dependencias con versiones exactas
├── README.md                        # Este archivo
├── datos/
│   ├── adult.zip                    # ZIP oficial descargado de UCI
│   ├── adult.data                   # 32 561 registros (partición de entrenamiento)
│   ├── adult.test                   # 16 281 registros (partición de prueba)
│   ├── adult.names                  # Descripción oficial de las variables
│   └── Index, old.adult.names       # Archivos que vienen dentro del ZIP oficial
└── modelos/
    ├── modelo_random_forest.joblib  # Pipeline completo: preprocesamiento + Random Forest
    ├── modelo_mlp.joblib            # Pipeline completo: preprocesamiento + MLP
    └── metadatos.json               # Orden de columnas, categorías válidas, rangos, métricas, versiones
```

**El preprocesamiento está dentro de cada `.joblib`** (imputación, `log1p`, escalado y one-hot ajustados con entrenamiento). No hay un archivo de preprocesamiento aparte: `joblib.load("modelos/modelo_mlp.joblib").predict(df)` recibe un DataFrame con las 12 columnas originales, en el orden de `metadatos.json["features"]`, y aplica todo internamente. `metadatos.json` solo lo usa la app para construir y validar los campos.

Las rutas son relativas: el notebook crea `datos/` y `modelos/` en su carpeta de trabajo y `app.py` busca `modelos/` junto a sí mismo (`Path(__file__).parent / "modelos"`), así que funciona desde cualquier ubicación siempre que se conserve esta estructura.

### Variaciones entrenadas

| Variación | Hiperparámetros | F1 val (CV 5) | Brecha train−val | Archivo | Decisión |
|---|---|---|---|---|---|
| RF-A · regularizado | 300 árboles, `max_depth=20`, `min_samples_leaf=5` | **0,664** | **0,017** | 8,2 MB | ✅ elegida |
| RF-B · árboles completos | 300 árboles, `max_depth=None`, `min_samples_leaf=1` | 0,655 | 0,295 | 53,2 MB | descartada |
| MLP-A · compacta + early stopping | capas (64, 32), `alpha=1e-3`, early stopping | **0,666** | **0,035** | 205 KB | ✅ elegida |
| MLP-B · amplia sin early stopping | capas (128, 64), `alpha=1e-4`, sin early stopping | 0,624 | 0,225 | 657 KB | descartada |

La selección se hizo solo con el 80 % de entrenamiento; el 20 % de prueba se reserva para comparar los dos modelos finales. 

### 3. La aplicación

Debajo de la portada, se explica qué predice la app, con qué datos y con qué modelos. Flujo guiado en 4 pasos: (1) elegir el modelo en la barra lateral, que incluye una breve introducción → (2) cargar un arquetipo de persona (Andrés, Laura, Carolina, Diego o perfil propio) → (3) ajustar los 12 campos → (4) Predecir. El resultado muestra la clase, la probabilidad, una barra con el umbral de 50 % y el nivel de confianza.

## 4. requirements.txt

```
# Versiones registradas al entrenar
streamlit>=1.36
scikit-learn==1.9.1
pandas==3.0.5
numpy==2.5.3
joblib==1.6.0
matplotlib==3.11.2
seaborn==0.13.2
```

`scikit-learn`, `numpy`, `pandas` y `joblib` van fijados porque el modelo guardado con joblib debe cargarse con la misma versión de scikit-learn con que se entrenó.

## 5. Ejecución

### A. Google Colab 

https://colab.research.google.com/drive/1IFeoUqlld3_K1EIHfb48IyNufghlMQ5h

### B. StreamLit

Ingresar https://predictor-nivel-ingreso.streamlit.app/
