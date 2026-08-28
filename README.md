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
| 1. Comprensión del negocio | [`01_comprension_del_negocio.ipynb`](notebooks/01_comprension_del_negocio.ipynb) | Plantilla con TODOs — contexto inicial redactado, objetivos y criterios de éxito por completar |
| 2. Comprensión de los datos | [`02_comprension_de_los_datos.ipynb`](notebooks/02_comprension_de_los_datos.ipynb) | Completo (carga, nulos, duplicados, hallazgos) reorganizado desde el notebook original; **faltan visualizaciones** (ver TODOs dentro del notebook) |
| 3. Preparación de los datos | [`03_preparacion_de_los_datos.ipynb`](notebooks/03_preparacion_de_los_datos.ipynb) | Pendiente — plantilla con TODOs |
| 4. Modelado | [`04_modelado.ipynb`](notebooks/04_modelado.ipynb) | Pendiente — plantilla con TODOs |
| 5. Evaluación | [`05_evaluacion.ipynb`](notebooks/05_evaluacion.ipynb) | Pendiente — plantilla con TODOs |
| 6. Interpretación y resultados | [`06_interpretacion_y_resultados.ipynb`](notebooks/06_interpretacion_y_resultados.ipynb) | Pendiente — plantilla con TODOs |

El notebook original íntegro (sin reorganizar) se conserva en [`notebooks/_original/`](notebooks/_original/) como respaldo.

## Estructura del repositorio

```
.
├── config/
│   └── config.yaml          # rutas, semilla (42), variable objetivo, hiperparámetros
├── data/
│   ├── raw/                 # datos originales de CQI, sin modificar
│   ├── interim/              # datos intermedios (aún vacío, TODO fase 03)
│   ├── processed/            # dataset final listo para modelar (aún vacío, TODO fase 03)
│   └── external/             # datos externos adicionales, si se agregan
├── models/                   # modelos serializados + metadatos (aún vacío, TODO fase 04)
├── notebooks/
│   ├── _original/             # respaldo del notebook original, sin editar
│   ├── 01_comprension_del_negocio.ipynb
│   ├── 02_comprension_de_los_datos.ipynb
│   ├── 03_preparacion_de_los_datos.ipynb
│   ├── 04_modelado.ipynb
│   ├── 05_evaluacion.ipynb
│   └── 06_interpretacion_y_resultados.ipynb
├── reports/
│   ├── figures/               # gráficas exportadas (.png) (aún vacío, TODO fase 02/05)
│   ├── tables/                # tablas de métricas (.csv) (aún vacío, TODO fase 04/05)
│   └── dashboard/             # datasets planos para el dashboard Power BI/Tableau (aún vacío, TODO fase 06)
├── src/
│   ├── config.py               # carga config/config.yaml
│   ├── utils.py                 # utilidades genéricas (ej. resumen de nulos)
│   ├── data/
│   │   ├── load_data.py          # carga de datos crudos
│   │   └── clean_data.py         # limpieza (TODO fase 03)
│   ├── features/
│   │   └── build_features.py     # ingeniería de variables (TODO fase 03)
│   ├── models/
│   │   ├── train_model.py        # entrenamiento (TODO fase 04)
│   │   ├── predict_model.py      # inferencia (TODO fase 04/05)
│   │   └── evaluate_model.py     # métricas (TODO fase 05)
│   └── visualization/
│       └── plots.py              # gráficas reutilizables (TODO fase 02/05)
├── README.md
├── requirements.txt
└── .gitignore
```

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
# 03-06: pendientes de implementación (ver TODOs en cada notebook)
```

Los datos crudos (`data/raw/*.csv`) ya están incluidos en el repositorio (son livianos y públicos), por lo que no se requiere ningún paso manual de descarga para ejecutar el notebook 02.

## Resultados principales

**TODO** — no hay ningún modelo entrenado todavía, por lo tanto no hay métricas reales que reportar. Esta tabla debe completarse en la fase 05 (`05_evaluacion.ipynb`) con los resultados reales obtenidos, y **no debe rellenarse con números de ejemplo**:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## Dashboard

El dashboard (Power BI o Tableau) se construye por fuera de este repositorio. Este repositorio solo debe dejar listos, en [`reports/dashboard/`](reports/dashboard/), los datasets planos (CSV) que lo alimentan — esa exportación está pendiente de implementarse en `06_interpretacion_y_resultados.ipynb`.

> TODO: una vez publicado el dashboard, agregar aquí el enlace/captura y describir brevemente qué vistas contiene.

## Limitaciones y trabajo futuro

- El dataset combina arabica (1311 filas) y robusta (28 filas) con un desbalance severo (~47:1); la estrategia para tratarlo (modelar por separado, submuestreo, sobremuestreo con SMOTE) aún no se ha decidido.
- Se detectó un registro en arabica con `Total.Cup.Points == 0`, muy probablemente un dato corrupto o un lote descalificado; debe tratarse explícitamente antes de construir la variable objetivo.
- Varias columnas tienen alta proporción de nulos (`Lot.Number` ~79%, `Variety` ~89% en robusta, entre otras) y varias variables categóricas están poco estandarizadas — la limpieza requiere decisiones cuidadosas, señaladas por el propio equipo como el mayor riesgo del proyecto.
- El pipeline de modelado, evaluación e interpretación (fases 03-06) todavía no está implementado; ver los TODOs dentro de cada notebook y de `src/`.

## Autores

TODO: completar con los nombres de los integrantes del grupo (máximo 3, según los lineamientos de la actividad).
