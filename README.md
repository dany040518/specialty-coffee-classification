# Clasificación de café de especialidad (Coffee Quality Institute)

Pipeline CRISP-DM para predecir si un lote de café califica como **"café de especialidad"** a partir de sus atributos de catación y origen, usando la base pública de la Coffee Quality Institute (CQI).

## Problema

El café es un producto de gran relevancia económica y social en Colombia. La Coffee Quality Institute mantiene una base de datos pública con más de 1300 lotes de café (arabica y robusta) evaluados profesionalmente bajo el protocolo de catación de la Specialty Coffee Association (SCA), que asigna puntajes a atributos como aroma, sabor, acidez, cuerpo y balance, y los combina en un puntaje total (`Total.Cup.Points`).

Este proyecto busca **predecir si un lote califica como "café de especialidad"** y **entender qué condiciones (origen, altitud, variedad, método de procesamiento, etc.) se asocian con esa clasificación**, con miras a que ese análisis pueda usarse para orientar al sector cafetero colombiano. La principal dificultad anticipada es que buena parte de las variables del dataset son categóricas, algunas con muchos valores únicos o poco estandarizados, lo que exige decisiones de limpieza cuidadosas.

## Objetivos

**General:** clasificar lotes de café como "café de especialidad" o no, a partir de sus atributos de catación y origen, e identificar las condiciones asociadas a esa clasificación.

**Específicos:**
- Construir y evaluar un modelo de clasificación sobre la base CQI (arabica/robusta).
- Identificar qué variables se asocian más fuertemente con la clasificación de especialidad, como insumo para recomendaciones al sector cafetero colombiano.
- TODO: el equipo debe completar/ajustar estos objetivos específicos en [`notebooks/01_comprension_del_negocio.ipynb`](notebooks/01_comprension_del_negocio.ipynb).

## Datos

- **Fuente:** [Coffee Quality Institute database](https://github.com/jldbc/coffee-quality-database) (repositorio público de James LeDoux).
- **Tamaño:** `arabica_data_cleaned.csv` (1311 filas × 44 columnas) y `robusta_data_cleaned.csv` (28 filas × 44 columnas). Desbalance severo entre especies (~47:1).
- **Variables principales:** subpuntajes de catación SCA (`Aroma`, `Flavor`, `Aftertaste`, `Acidity`, `Body`, `Balance`, `Uniformity`, `Clean.Cup`, `Sweetness`, `Cupper.Points`), puntaje total (`Total.Cup.Points`), y variables de contexto (`Country.of.Origin`, `Variety`, `Processing.Method`, `Altitude`/`altitude_mean_meters`, `Color`, `Harvest.Year`, entre otras).
- **Variable objetivo:** **TODO** — aún no está construida en el código. El candidato natural es una variable binaria derivada de `Total.Cup.Points` usando el umbral estándar SCA (`>= 80` = café de especialidad), pero esto debe confirmarse y documentarse en la fase de preparación de datos (ver hallazgo sobre un registro con `Total.Cup.Points == 0` en el notebook 02, que debe tratarse antes de aplicar cualquier umbral).

## Metodología CRISP-DM

| Fase | Notebook | Estado |
|---|---|---|
| 1. Comprensión del negocio | [`01_comprension_del_negocio.ipynb`](notebooks/01_comprension_del_negocio.ipynb) | Contexto redactado — objetivos específicos y criterios de éxito quedan como `TODO (autor)`: son una decisión del equipo, no algo que deba fabricarse |
| 2. Comprensión de los datos | [`02_comprension_de_los_datos.ipynb`](notebooks/02_comprension_de_los_datos.ipynb) | Completo, con visualizaciones (distribución del target, correlación de subpuntajes, países de origen) |
| 3. Preparación de los datos | [`03_preparacion_de_los_datos.ipynb`](notebooks/03_preparacion_de_los_datos.ipynb) | Completo — limpieza, variable objetivo, encoding/escalado y split train/test |
| 4. Modelado | [`04_modelado.ipynb`](notebooks/04_modelado.ipynb) | Completo — regresión logística (baseline) + SMOTE + validación cruzada |
| 5. Evaluación | [`05_evaluacion.ipynb`](notebooks/05_evaluacion.ipynb) | Completo — métricas, matriz de confusión, curva ROC y análisis de errores sobre test |
| 6. Interpretación y resultados | [`06_interpretacion_y_resultados.ipynb`](notebooks/06_interpretacion_y_resultados.ipynb) | Completo — interpretación de coeficientes, limitaciones, export a `reports/dashboard/` |

Toda la lógica de las fases 2-6 vive en `src/` (arquitectura por capas); los notebooks son narrativos y solo llaman a `src/pipeline.py`. El notebook original íntegro (sin reorganizar) se conserva en [`notebooks/_original/`](notebooks/_original/) como respaldo.

### Decisiones de modelado tomadas

- **Variable objetivo:** `especialidad = 1` si `Total.Cup.Points >= 80` (umbral estándar SCA), tras eliminar el único registro con puntaje corrupto (`Total.Cup.Points == 0`).
- **Arabica + Robusta:** combinados en un solo dataset (columna `species`); el desbalance de la variable objetivo (~86%/14%) se trata con SMOTE dentro del pipeline de entrenamiento.
- **Variables predictoras:** solo variables de contexto (origen, variedad, método de procesamiento, altitud, color, especie, tamaño del lote, defectos, humedad). Se excluyen deliberadamente los subpuntajes de catación (`Aroma`, `Flavor`, ...) porque son los componentes que suman `Total.Cup.Points` y usarlos filtraría el target.
- **Modelo:** regresión logística (baseline), interpretable vía coeficientes. `src/models/train_model.py::MODEL_REGISTRY` queda abierto para agregar Random Forest / XGBoost sin modificar el código de entrenamiento.

## Estructura del repositorio

```
.
├── config/
│   └── config.yaml            # rutas, semilla (42), variable objetivo, split, hiperparámetros
├── data/
│   ├── raw/                   # datos originales de CQI, sin modificar
│   ├── interim/                # sin uso en esta iteración (limpieza y target se resuelven en un solo paso)
│   ├── processed/              # coffee_processed.csv, train.csv, test.csv (generados por 03)
│   └── external/               # datos externos adicionales, si se agregan
├── models/                     # logistic_regression.joblib + metadata.json (generados por 04)
├── notebooks/
│   ├── _original/               # respaldo del notebook original, sin editar
│   ├── 01_comprension_del_negocio.ipynb
│   ├── 02_comprension_de_los_datos.ipynb
│   ├── 03_preparacion_de_los_datos.ipynb
│   ├── 04_modelado.ipynb
│   ├── 05_evaluacion.ipynb
│   └── 06_interpretacion_y_resultados.ipynb
├── reports/
│   ├── figures/                 # gráficas exportadas (.png) — EDA + evaluación + coeficientes
│   ├── tables/                  # métricas de test (.json)
│   └── dashboard/               # dataset plano (.csv) para el dashboard Power BI/Tableau
├── src/
│   ├── config.py                 # carga config/config.yaml
│   ├── utils.py                   # utilidades genéricas (ej. resumen de nulos)
│   ├── pipeline.py                # orquestación (fachada): prepare_data / train / evaluate
│   ├── data/
│   │   ├── base.py                  # interfaces DataLoader, DataCleaningStep
│   │   ├── load_data.py             # CQIRawDataLoader
│   │   └── clean_data.py            # DropDuplicateRows, DropCorruptScores, CleaningPipeline
│   ├── features/
│   │   ├── base.py                  # interfaz FeatureBuilder
│   │   └── build_features.py        # SpecialtyTargetBuilder, preprocesador, split train/test
│   ├── models/
│   │   ├── base.py                  # interfaces ModelTrainer, Predictor, Evaluator
│   │   ├── train_model.py           # SklearnModelTrainer (preprocesador+SMOTE+estimador), MODEL_REGISTRY
│   │   ├── predict_model.py         # ModelPredictor
│   │   └── evaluate_model.py        # ClassificationEvaluator
│   └── visualization/
│       └── plots.py                 # gráficas reutilizables (EDA + evaluación + coeficientes)
├── README.md
├── requirements.txt
└── .gitignore
```

**Arquitectura:** por capas (`data` → `features` → `models` → `visualization`), orquestadas por `src/pipeline.py`. Cada capa expone una interfaz (`base.py`) de la que dependen las capas superiores (Dependency Inversion), y los pasos de limpieza/modelos se registran de forma extensible (`CleaningPipeline`, `MODEL_REGISTRY`) sin modificar código existente (Open/Closed).

## Instrucciones de ejecución

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd <nombre-del-repositorio>

# 2. Crear y activar un entorno virtual
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar los notebooks en orden (cada uno lee lo que dejó el anterior)
jupyter notebook notebooks/01_comprension_del_negocio.ipynb
jupyter notebook notebooks/02_comprension_de_los_datos.ipynb
jupyter notebook notebooks/03_preparacion_de_los_datos.ipynb
jupyter notebook notebooks/04_modelado.ipynb
jupyter notebook notebooks/05_evaluacion.ipynb
jupyter notebook notebooks/06_interpretacion_y_resultados.ipynb

# También se pueden ejecutar de punta a punta sin abrir Jupyter:
# jupyter nbconvert --to notebook --execute --inplace notebooks/0{2,3,4,5,6}_*.ipynb
```

Los datos crudos (`data/raw/*.csv`) ya están incluidos en el repositorio (son livianos y públicos), por lo que no se requiere ningún paso manual de descarga para ejecutar el notebook 02.

## Resultados principales

Resultados reales de la regresión logística (baseline), sobre el conjunto de **test** (20%, nunca visto en entrenamiento ni validación cruzada). Generados por `notebooks/05_evaluacion.ipynb` / `src/models/evaluate_model.py`, guardados en [`reports/tables/logistic_regression_metrics.json`](reports/tables/logistic_regression_metrics.json):

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Regresión logística (baseline) | 0.687 | 0.940 | 0.680 | 0.789 | 0.760 |

**Lectura:** con precision alta (0.94) y recall más moderado (0.68), el modelo es conservador — cuando predice "especialidad" casi siempre acierta, pero deja pasar una parte de los lotes que sí lo son. Ver interpretación completa (coeficientes, países/variables más asociadas) en [`06_interpretacion_y_resultados.ipynb`](notebooks/06_interpretacion_y_resultados.ipynb).

## Dashboard

El dashboard (Power BI o Tableau) se construye por fuera de este repositorio. [`reports/dashboard/coffee_dashboard.csv`](reports/dashboard/coffee_dashboard.csv) ya contiene el dataset plano (variables de contexto + target real + predicción + probabilidad) que lo alimenta, generado por `06_interpretacion_y_resultados.ipynb`.

> TODO: una vez publicado el dashboard, agregar aquí el enlace/captura y describir brevemente qué vistas contiene.

## Limitaciones y trabajo futuro

- El dataset combina arabica (1311 filas) y robusta (28 filas) con un desbalance severo (~47:1) de especie; se combinaron en un solo dataset con `species` como variable, y el desbalance que se trató con SMOTE fue el de la variable objetivo (~86%/14%), no el de especie — ver limitaciones detalladas en `06_interpretacion_y_resultados.ipynb`.
- Se detectó y eliminó un registro en arabica con `Total.Cup.Points == 0` (dato corrupto), antes de construir la variable objetivo.
- Varias columnas tienen alta proporción de nulos (`Lot.Number` ~79%, `Variety` ~89% en robusta, entre otras); se imputaron como mediana (numéricas) o `"Unknown"` (categóricas) en vez de descartarse.
- Solo se entrenó un modelo (regresión logística). `src/models/train_model.py::MODEL_REGISTRY` queda listo para agregar Random Forest / XGBoost sin modificar el pipeline de entrenamiento — ver TODOs en `04_modelado.ipynb`.
- Los objetivos específicos y los criterios de éxito de `01_comprension_del_negocio.ipynb` quedan como decisión pendiente del equipo (`TODO (autor)`); una vez definidos, `05_evaluacion.ipynb` debe contrastarlos explícitamente contra las métricas de test.

## Autores
Mariana Valle Moreno
Danna Alejandra Sanchez Monsalve