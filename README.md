# Perfiles de calidad de café mediante clustering (Coffee Quality Institute)

Proyecto CRISP-DM de aprendizaje **no supervisado**: agrupar lotes de café por sus
atributos de catación SCA para descubrir **perfiles de calidad** y luego
caracterizar esos grupos según el origen y el manejo del cultivo (país, altitud,
variedad, método de procesamiento). Usa la base pública de la Coffee Quality
Institute (CQI).

## Problema

El café es una de las actividades más importantes del campo colombiano y sostiene
a cientos de miles de familias caficultoras. La calidad de un lote se mide con el
protocolo de catación de la Specialty Coffee Association (SCA): un panel evalúa
aroma, sabor, acidez, cuerpo, balance y otros atributos, y los combina en un
puntaje sobre 100; por encima de 80 el lote se considera "café de especialidad".

Ese puntaje solo se conoce al final, cuando el café ya fue cosechado y catado.
En lugar de imponer desde el inicio un único umbral, este proyecto deja que los
**datos revelen si existen distintos perfiles de café** y qué los caracteriza, con
la idea de que ese entendimiento sirva de guía práctica al sector cafetero a la
hora de decidir qué variedad sembrar o cómo estandarizar el manejo poscosecha.

## Objetivos

**General:** explorar, mediante clustering, si los lotes de café pueden agruparse
en distintos perfiles de calidad y, a partir de esos grupos, entender qué
características de cultivo y manejo (origen, variedad, procesamiento) se asocian
con los lotes que mejor califican.

**Específicos:**
- Encontrar grupos naturales entre los lotes a partir de sus atributos en la taza
  (aroma, sabor, aftertaste, acidez, cuerpo, balance, puntaje del catador),
  probando varias representaciones y algoritmos hasta quedarse con el que mejor
  refleje esas diferencias.
- Revisar qué tienen en común los lotes de cada grupo en cuanto a país, altitud,
  variedad y procesamiento, para distinguir a los que mejor califican.
- Mostrar los grupos y lo que los caracteriza en un dashboard pensado para una
  audiencia no técnica.

Ver detalle en [`notebooks/01_comprension_del_negocio.ipynb`](notebooks/01_comprension_del_negocio.ipynb).

## Datos

- **Fuente:** [Coffee Quality Institute database](https://github.com/jldbc/coffee-quality-database) (repositorio público de James LeDoux).
- **Tamaño original:** `arabica_data_cleaned.csv` (1311 filas × 44 columnas) y `robusta_data_cleaned.csv` (28 filas × 44 columnas). Ambos están incluidos en `data/raw/` (son livianos y públicos), así que no hace falta ningún paso de descarga.
- **Alcance:** se trabaja **solo con arabica**. Robusta se descartó en la fase 3 porque usa otro formulario de catación (*Fine Robusta*, con atributos que no existen en arabica), tiene alta nulidad en las variables de perfilado (`Variety` ~89 %, `Processing.Method` ~64 %, `Moisture` ~43 % en 0) y solo aporta 28 registros.
- **Variables que entran al clustering** (subpuntajes de catación SCA): `Aroma`, `Flavor`, `Aftertaste`, `Acidity`, `Body`, `Balance`, `Cupper.Points`. Se dejaron fuera `Uniformity`, `Clean.Cup` y `Sweetness` por ser casi constantes en 10, y `Total.Cup.Points` por ser la suma de los subpuntajes.
- **Variables para caracterizar los grupos** (no entran al clustering): `Country.of.Origin`, `altitude_mean_meters`, `Variety`, `Processing.Method`, `Color`, `Harvest.Year`, defectos y `Quakers`.
- **No hay variable objetivo.** `Total.Cup.Points` se conserva aparte y solo se usa como referencia para contrastar los grupos contra la convención de la industria (`>= 80` = café de especialidad).

## Metodología CRISP-DM

| Fase | Notebook | Estado |
|---|---|---|
| 1. Comprensión del negocio | [`01_comprension_del_negocio.ipynb`](notebooks/01_comprension_del_negocio.ipynb) | Completo — problema, objetivos, evaluación de la situación, objetivos de minería de datos y criterios de éxito. |
| 2. Comprensión de los datos | [`02_comprension_de_los_datos.ipynb`](notebooks/02_comprension_de_los_datos.ipynb) | Completo — descripción de ambos datasets, calidad (nulos y duplicados), descriptivas y atípicos de las cuantitativas, frecuencias de las cualitativas, y conclusiones sobre qué tratar en la fase 3. |
| 3. Preparación de los datos | [`03_preparacion_de_los_datos.ipynb`](notebooks/03_preparacion_de_los_datos.ipynb) | Completo — limpieza paso a paso, decisión de alcance (solo arabica), reducción de variables, análisis de redundancia (correlación y V de Cramér), imputación de perfilado y construcción de la matriz de entrada estandarizada. |
| 4. Modelado | [`04_modelado.ipynb`](notebooks/04_modelado.ipynb) | Completo — comparación de representaciones (7 atributos vs. PCA) × algoritmos (k-means, aglomerativo, gaussian mixture) × número de grupos, y ajuste del modelo final. |
| 5. Evaluación | [`05_evaluacion.ipynb`](notebooks/05_evaluacion.ipynb) | **Pendiente** — validación interna, estabilidad (ARI con resiembras y submuestras) y contraste externo contra `Total.Cup.Points >= 80`. |
| 6. Interpretación y resultados | [`06_interpretacion_y_resultados.ipynb`](notebooks/06_interpretacion_y_resultados.ipynb) | **Pendiente** — perfilado de cada grupo por origen y manejo, limitaciones y export a `reports/dashboard/`. |

Cada notebook lee lo que dejó el anterior. La lógica reutilizable vive en `src/`
(funciones planas, sin clases base ni jerarquías), orquestada por
`src/pipeline.py`; los parámetros (semilla, rutas, columnas, algoritmos, modelo
elegido) salen de `config/config.yaml`, no de valores fijos en los notebooks.

### Decisiones tomadas hasta ahora

- **Registro corrupto:** se eliminó la única fila con `Total.Cup.Points == 0` (todos los subpuntajes en 0). El dataset de trabajo queda en **1310 filas**.
- **Columnas:** se descartaron 23 columnas entre identificadores administrativos con alta nulidad, columnas de altitud redundantes / de texto libre, y datos del proceso de certificación que no describen el café. El dataset reducido tiene 23 columnas (`data/processed/coffee_clean.csv`).
- **Valores imposibles:** `altitude_mean_meters` fuera de `[200, 3000]` m → `NaN`; `Moisture == 0` → `NaN` (la humedad del grano nunca es 0 %).
- **Imputación:** las variables de perfilado categóricas se rellenan con `"Desconocido"` y se agregan banderas `altitude_reportada` / `humedad_reportada`; no se imputa la altitud con la mediana porque los faltantes no son aleatorios (MNAR).
- **Matriz de clustering:** los 7 subpuntajes SCA estandarizados (media 0, desviación 1) → `data/processed/clustering_input.csv` (1310 × 7).
- **Reducción por PCA:** la primera componente concentra ~72 % de la varianza y recoge un eje de "calidad general" (los 7 atributos cargan fuerte y en la misma dirección). Se comparan tres representaciones: los 7 atributos, `pca_2` (2 componentes) y `pca_4` (~90 % de la varianza).
- **Modelo final:** `pca_2` + **k-means** con **k = 3** grupos (`config -> clustering.final`). Se elige k-means sobre el aglomerativo (resultado casi idéntico, más simple y reproducible) y k = 3 sobre k = 2 (k = 2 solo parte el eje de calidad en "mejor/peor", que no aporta frente a lo que ya dice el umbral de 80 puntos).

## Estructura del repositorio

```
.
├── config/
│   └── config.yaml            # semilla (42), rutas, columnas, algoritmos, modelo elegido
├── data/
│   ├── raw/                   # datos originales de CQI, sin modificar
│   ├── interim/               # sin uso en esta iteración
│   ├── processed/             # clustering_input.csv, coffee_clean.csv, clustering_scaler.json,
│   │                          #   coffee_clustered.csv  (generados por 03 y 04)
│   └── external/              # sin uso
├── models/                    # clustering_model.joblib + clustering_metadata.json (generados por 04)
├── notebooks/
│   ├── 01_comprension_del_negocio.ipynb
│   ├── 02_comprension_de_los_datos.ipynb
│   ├── 03_preparacion_de_los_datos.ipynb
│   ├── 04_modelado.ipynb
│   ├── 05_evaluacion.ipynb                 # pendiente
│   └── 06_interpretacion_y_resultados.ipynb  # pendiente
├── reports/
│   ├── figures/               # gráficas .png — EDA (fase 02) + PCA y clustering (fase 04)
│   ├── tables/                # clustering_comparison.csv (todas las combinaciones probadas)
│   └── dashboard/             # dataset plano para Power BI / Tableau (pendiente, fase 06)
├── src/
│   ├── config.py             # carga config/config.yaml y resuelve rutas relativas al repo
│   ├── utils.py              # utilidades genéricas (p. ej. resumen de nulos)
│   ├── pipeline.py           # fachada: prepare_data (fase 03) y train_clustering (fase 04)
│   ├── data/
│   │   ├── load_data.py      # carga de los CSV crudos de CQI
│   │   └── clean_data.py     # limpieza: registro corrupto, columnas, altitud, Moisture, texto
│   ├── features/
│   │   └── build_features.py # imputación de perfilado y matriz de clustering estandarizada
│   ├── models/
│   │   ├── train_model.py    # representaciones, barrido rep × algoritmo × k, modelo final
│   │   ├── evaluate_model.py # validación y estabilidad de los grupos (pendiente, fase 05)
│   │   └── predict_model.py  # asignación de lotes nuevos a un grupo (pendiente, fase 05/06)
│   └── visualization/
│       └── plots.py          # gráficas reutilizables de EDA
├── README.md
├── requirements.txt
└── .gitignore
```

**Arquitectura:** capas `data → features → models → visualization` como módulos de
funciones planas (sin `base.py` ni ABCs), orquestadas por `src/pipeline.py`.
`prepare_data(config)` reproduce toda la fase 03 y `train_clustering(config)` toda
la fase 04, cada una en una sola llamada y con los mismos artefactos que producen
los notebooks.

## Instrucciones de ejecución

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd specialty-coffee-classification

# 2. Crear y activar un entorno virtual
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar los notebooks en orden (cada uno lee lo que dejó el anterior)
jupyter notebook notebooks/01_comprension_del_negocio.ipynb
jupyter notebook notebooks/02_comprension_de_los_datos.ipynb
jupyter notebook notebooks/03_preparacion_de_los_datos.ipynb
jupyter notebook notebooks/04_modelado.ipynb

# También, de punta a punta y sin abrir Jupyter:
# jupyter nbconvert --to notebook --execute --inplace notebooks/0{2,3,4}_*.ipynb
```

Las fases 03 y 04 también se pueden correr desde código:

```python
from src.config import load_config
from src.pipeline import prepare_data, train_clustering

config = load_config()
prepare_data(config)       # -> data/processed/
train_clustering(config)   # -> models/, data/processed/coffee_clustered.csv, reports/tables/
```

## Resultados preliminares (fase 04)

Modelo final: PCA a 2 dimensiones + k-means con k = 3, sobre los 1310 lotes de
arabica. Métricas internas (`models/clustering_metadata.json`):

| Métrica | Valor | Lectura |
|---|---|---|
| Coeficiente de silueta | 0.428 | Separación moderada. El espacio sensorial está dominado por un solo eje (calidad general), así que los grupos son tres tramos de un continuo más que nubes aisladas. |
| Davies-Bouldin | 0.761 | — (más bajo es mejor) |
| Calinski-Harabasz | 1506 | — (más alto es mejor) |

Perfil de los tres grupos (los nombres son descriptivos, a confirmar en la fase 06):

| Grupo | n | `Total.Cup.Points` medio | Altitud media (m) | % ≥ 80 pts | Orígenes más frecuentes |
|---|---|---|---|---|---|
| 2 — calidad alta | 307 | 84.8 | 1478 | 99 % | Colombia, Etiopía, Guatemala |
| 0 — calidad media | 708 | 82.4 | 1344 | 97 % | Colombia, México, Guatemala |
| 1 — calidad más baja | 295 | 78.9 | 1230 | 48 % | México, Guatemala, Taiwán |

> La validación de estabilidad (fase 05) y el perfilado detallado por origen y
> manejo (fase 06) están pendientes; estos números son la salida directa del
> modelado y pueden ajustarse.

## Dashboard

El dashboard (Power BI o Tableau) se construye por fuera de este repositorio y se
alimentará de un dataset plano (grupo de cada lote + sus características) que
generará la fase 06 en `reports/dashboard/`.

> TODO: una vez publicado, agregar aquí el enlace/captura y describir qué vistas contiene.

## Limitaciones y trabajo futuro

- La base de datos solo incluye cafés que buscaron una certificación de calidad de CQI, así que no representa a todo el café que se produce; el 86 % de los lotes ya supera los 80 puntos.
- Se trabajó solo con arabica; robusta quedó fuera por formulario de catación distinto, alta nulidad y tamaño (28 filas).
- La separación entre grupos es moderada (silueta ≈ 0.43) porque los 7 atributos sensoriales están muy correlacionados: casi toda la variación cabe en un eje de calidad general. El agrupamiento se usa como base **descriptiva**, no como una frontera nítida.
- Hay países y variedades con muy pocos lotes; las conclusiones sobre ellos en la fase 06 deben tomarse con cautela.
- Faltan las fases 05 (evaluación y estabilidad) y 06 (interpretación y dashboard).

## Autoras

Mariana Valle Moreno
Danna Alejandra Sanchez Monsalve
