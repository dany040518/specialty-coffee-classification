# Perfiles de calidad de café mediante clustering (Coffee Quality Institute)

Proyecto CRISP-DM de aprendizaje **no supervisado**: agrupar lotes de café por
sus atributos de catación SCA para descubrir **perfiles de calidad**, y luego
ver qué los distingue en origen y manejo (país, altitud, variedad,
procesamiento). Usa la base pública del Coffee Quality Institute (CQI).

## Problema

El café sostiene a cientos de miles de familias caficultoras. Su calidad se
mide con el protocolo de catación de la SCA: un panel evalúa aroma, sabor,
acidez, cuerpo y balance, y los combina en un puntaje sobre 100; por encima de
80 el lote se considera "café de especialidad".

Ese puntaje solo se conoce al final, cuando el café ya fue cosechado y
catado. En vez de partir de un umbral fijo, este proyecto deja que los datos
digan si existen perfiles distintos de café y qué los caracteriza — con la
idea de que eso sirva de guía al sector cafetero para decidir qué sembrar o
cómo estandarizar el manejo poscosecha.

## Objetivos

**General:** explorar, mediante clustering, si los lotes de café se agrupan en
perfiles de calidad distintos y qué características de cultivo y manejo se
asocian con los que mejor califican.

**Específicos:**
- Encontrar grupos naturales a partir de los atributos en taza (aroma, sabor,
  aftertaste, acidez, cuerpo, balance, puntaje del catador), probando varias
  representaciones y algoritmos.
- Ver qué tienen en común los lotes de cada grupo en país, altitud, variedad y
  procesamiento.
- Mostrar los grupos en un dashboard pensado para una audiencia no técnica.

Detalle en [`01_comprension_del_negocio.ipynb`](notebooks/01_comprension_del_negocio.ipynb).

## Datos

- **Fuente:** [Coffee Quality Institute database](https://github.com/jldbc/coffee-quality-database), de James LeDoux.
- **Tamaño original:** `arabica_data_cleaned.csv` (1311 × 44) y `robusta_data_cleaned.csv` (28 × 44), incluidos en `data/raw/` — livianos, públicos, sin paso de descarga.
- **Alcance: solo arabica.** Robusta se descartó en la fase 3: usa otro formulario de catación (*Fine Robusta*), tiene alta nulidad en las variables de perfilado (`Variety` ~89 %, `Processing.Method` ~64 %, `Moisture` ~43 % en 0) y aporta solo 28 registros.
- **Entran al clustering** (subpuntajes SCA): `Aroma`, `Flavor`, `Aftertaste`, `Acidity`, `Body`, `Balance`, `Cupper.Points`. Quedaron fuera `Uniformity`, `Clean.Cup` y `Sweetness` por ser casi constantes en 10, y `Total.Cup.Points` por ser la suma de los demás.
- **Caracterizan los grupos, sin entrar al clustering:** `Country.of.Origin`, `altitude_mean_meters`, `Variety`, `Processing.Method`, `Color`, `Harvest.Year`, defectos y `Quakers`.
- **No hay variable objetivo.** `Total.Cup.Points` se guarda aparte, solo como referencia para contrastar los grupos contra el umbral de la industria (`>= 80`).

## Metodología CRISP-DM

| Fase | Notebook | Estado |
|---|---|---|
| 1. Comprensión del negocio | [`01_comprension_del_negocio.ipynb`](notebooks/01_comprension_del_negocio.ipynb) | Completo — problema, objetivos, criterios de éxito. |
| 2. Comprensión de los datos | [`02_comprension_de_los_datos.ipynb`](notebooks/02_comprension_de_los_datos.ipynb) | Completo — calidad de ambos datasets, descriptivas, atípicos, frecuencias, y qué tratar en la fase 3. |
| 3. Preparación de los datos | [`03_preparacion_de_los_datos.ipynb`](notebooks/03_preparacion_de_los_datos.ipynb) | Completo — limpieza, alcance (solo arabica), reducción de variables, redundancia (correlación y V de Cramér), imputación y matriz de entrada estandarizada. |
| 4. Modelado | [`04_modelado.ipynb`](notebooks/04_modelado.ipynb) | Completo — representaciones (7 atributos vs. PCA) × algoritmos (k-means, aglomerativo, gaussian mixture) × k, y ajuste del modelo final. |
| 5. Evaluación | [`05_evaluacion.ipynb`](notebooks/05_evaluacion.ipynb) | Completo — reproducibilidad, validación interna, estabilidad (semillas y submuestras), contraste externo, y por qué k = 3 y no k = 2. |
| 6. Interpretación y resultados | [`06_interpretacion_y_resultados.ipynb`](notebooks/06_interpretacion_y_resultados.ipynb) | Completo — perfilado por país, variedad, procesamiento, altitud y defectos; descripción en lenguaje llano de los tres grupos; export a `reports/dashboard/`. |

Cada notebook lee lo que dejó el anterior. La lógica reutilizable vive en
`src/` (funciones planas, sin clases base), orquestada por `src/pipeline.py`;
los parámetros (semilla, rutas, columnas, algoritmos, modelo elegido) salen de
`config/config.yaml`, nunca de valores fijos en los notebooks.

### Decisiones de preparación (fase 03)

- **Registro corrupto:** se eliminó la única fila con `Total.Cup.Points == 0` (todos los subpuntajes en 0). Quedan **1310 filas**.
- **Columnas:** se descartaron 23 (identificadores administrativos con alta nulidad, altitud redundante o en texto libre, datos de certificación). El dataset reducido queda en 23 columnas (`data/processed/coffee_clean.csv`).
- **Valores imposibles:** `altitude_mean_meters` fuera de `[200, 3000]` m → `NaN`; `Moisture == 0` → `NaN` (la humedad del grano nunca es 0 %).
- **Imputación:** las categóricas de perfilado se rellenan con `"Desconocido"`, con banderas `altitude_reportada` / `humedad_reportada`. La altitud no se imputa con la mediana porque sus faltantes no son aleatorios (MNAR).
- **Matriz de clustering:** los 7 subpuntajes estandarizados (media 0, desviación 1) → `data/processed/clustering_input.csv` (1310 × 7).

## El modelo (fase 04)

PCA a 2 dimensiones + k-means con **k = 3**, sobre los 1310 lotes, semilla 42.
Se compararon tres representaciones (7 atributos, `pca_2`, `pca_4`, esta última
con ~90 % de la varianza) y tres algoritmos; k-means sobre `pca_2` ganó por dar
un resultado casi idéntico al aglomerativo pero más simple y reproducible.

| Métrica | Valor | Lectura |
|---|---|---|
| Silueta | 0.428 | Separación moderada: un solo eje de "calidad general" domina el espacio, así que los grupos son tramos de un continuo más que nubes aisladas. |
| Davies-Bouldin | 0.761 | Más bajo es mejor. |
| Calinski-Harabasz | 1506 | Más alto es mejor. |

Los grupos quedan en 708, 295 y 307 lotes (qué contiene cada uno, en la
sección de resultados más abajo). Barrido completo en
[`clustering_comparison.csv`](reports/tables/clustering_comparison.csv).

**Por qué k = 3 y no k = 2.** La fase 4 había descartado k = 2 asumiendo que
solo repetía el umbral de 80 puntos. La fase 5 lo midió y no es así: el ARI de
k = 2 contra la binaria de 80 puntos es 0.155, más bajo incluso que el de k = 3
(0.177) — ninguno reproduce la convención de la industria. Lo que sostiene k = 3
es que aísla en un grupo a 154 de los 180 lotes bajo 80 puntos y separa además
dos tramos distinguibles por encima del umbral.

## Resultados de la evaluación (fase 05)

Tres vías de validación: métricas internas, estabilidad y una referencia
externa que no entró al agrupamiento (`models/clustering_evaluation.json`,
`reports/tables/`).

| Prueba | Resultado | Lectura |
|---|---|---|
| Reproducibilidad | ARI = **1.0** contra las etiquetas de la fase 04 | El modelo se reconstruye desde cero y da la misma partición. |
| Silueta en `pca_2` | **0.428** | La misma de la fase 04. |
| Silueta en los 7 atributos | **0.296** | En el espacio donde los grupos se interpretan, la separación cae casi a la mitad — reducir dimensiones sube la silueta por construcción. Davies-Bouldin pasa de 0.761 a 1.088 y Calinski-Harabasz de 1506 a 812. |
| Estabilidad ante la semilla | ARI medio **0.996** (20 semillas) | 13 de 20 corridas dan una partición idéntica. |
| Estabilidad ante submuestras | ARI medio **0.952** (50 submuestras del 80 %) | Los grupos sobreviven a quitar un quinto de los lotes; el peor caso da 0.777. |
| Estabilidad de la asignación | **98.4 %** conserva su grupo | La variabilidad se concentra en unos 32 lotes de frontera. |
| Contraste externo | ARI **0.177** contra `Total.Cup.Points >= 80` | Los grupos se ordenan por puntaje (78.9, 82.4, 84.8 de promedio) sin haberlo visto, pero no reproducen el corte de la industria: el umbral parte al grupo 1 casi por la mitad. |
| PCA guardado | `models/clustering_pca.joblib` | La fase 4 había guardado el k-means pero no el PCA sobre el que se ajustó; sin él no se podía proyectar un lote nuevo. Se reconstruye acá. |

Contra los criterios de éxito de la fase 1: el criterio de minería se cumple
en estabilidad e interpretabilidad, y se cumple a medias en diferenciación
(silueta 0.43 es moderada, no excelente). El criterio de negocio depende de la
fase 06.

## Los tres grupos de cafés (fase 06)

| Grupo | Nombre | n | Puntaje medio | % ≥ 80 | Altitud media | Origen principal | Rasgo que lo distingue |
|---|---|---|---|---|---|---|---|
| 0 | Cafés de puntaje medio, lavados y latinoamericanos | 708 | 82.4 | 96.6 % | 1344 m | Colombia 18 %, México 17 %, Guatemala 14 % | El grueso de la base; tan limpio como el grupo 2 pero medio punto menos en todos los atributos. |
| 1 | Cafés de puntaje más bajo, de fincas bajas y con más defectos | 295 | 78.9 | 47.8 % | 1230 m | México 32 %, Guatemala 16 % | El único donde la mayoría no llega a 80 puntos, y el único con más defectos: 5.71 de categoría dos por lote contra 3.09 y 2.72. |
| 2 | Cafés de puntaje alto, de fincas altas y orígenes variados | 307 | 84.8 | 99.3 % | 1478 m | Colombia 16 %, Etiopía 12 %, Guatemala 12 % | El más internacional y de fincas más altas; menos lavado (52 %) y más natural (24 %) que los demás. |

Los tres grupos quedan ordenados igual en los siete atributos de catación, sin
excepción: se separan por nivel general de la taza, no por un perfil de sabor
distinto. Detalle en la sección 6.9 del notebook 06 y en
`reports/tables/perfilado_*.csv`.

## Dashboard

Se construye por fuera de este repositorio, a partir de
[`coffee_dashboard_data.csv`](reports/dashboard/coffee_dashboard_data.csv):
**1310 filas** × 27 columnas, con origen, manejo, el grupo de cada lote, su
nombre descriptivo (`grupo_nombre`), la marca de especialidad
(`especialidad`) y el tramo de puntaje (`rango_puntaje`).

Vistas sugeridas (notebook 06, sección 6.12): mapa por país (tamaño = cantidad
de lotes, color = grupo predominante); altitud contra puntaje coloreado por
grupo, para ver la tendencia y el solape a la vez; barras apiladas de
procesamiento y variedad por grupo; y una tarjeta por grupo con nombre,
tamaño, puntaje promedio y % de especialidad.

> Cuando el tablero esté publicado va acá el enlace, con captura y descripción de sus vistas.

## Limitaciones y consideraciones

La base del CQI reúne cafés que se enviaron a certificar, no una muestra del
café que se produce en el mundo: el 86.3 % de los lotes ya superaba los 80
puntos antes de empezar. Por eso el grupo de menor puntaje no es "café malo",
es el tramo más bajo de un conjunto que ya partía por encima del promedio —
casi la mitad de sus lotes igual califica como especialidad. Y todo esto vale
solo para arabica; robusta quedó fuera desde la fase 03.

Que el grupo de mejor puntaje tenga más Colombia, más Etiopía y 250 metros
más de altura promedio es una diferencia medible, no una prueba de que
sembrar ahí produzca mejor café: son lotes reales, con su suelo, su clima, su
manejo y su catador, todo junto y sin separar. Los rangos de altitud, de
hecho, se solapan casi por completo entre los tres grupos — esto es un mapa
de qué se parece a qué, no una regla de "siembre por encima de tal altura".

Los siete atributos de catación están tan correlacionados que casi toda la
variación cabe en un solo eje de calidad general, así que los grupos son
tramos de un continuo y no nubes separadas — de ahí la silueta moderada (0.43,
y 0.30 en el espacio de los siete atributos) y que unos 32 lotes de frontera
cambien de grupo según la muestra. Son estables (ARI 0.996 al cambiar la
semilla, 0.952 con submuestras), lo que es distinto de decir que tres sea el
número "verdadero" de perfiles de café: es una descripción repetible con
estos datos y este algoritmo, mejor leída como niveles de calidad que como
categorías cerradas. Tampoco equivalen a la categoría de especialidad — el
corte de 80 puntos no coincide con ninguna frontera del modelo (ARI 0.18).

Por último: hay países y variedades con muy pocos lotes, así que cualquier
lectura sobre ellos —sobre todo en el tablero— debería mostrar el `n` al lado
del porcentaje.

## Trabajo pendiente

- Construir el tablero en Power BI o Tableau y enlazarlo acá — la única exigencia de la fase 01 que sigue abierta.
- Preparar la sustentación con las descripciones de 6.9 y la tabla de criterios de 6.10.
- Si el proyecto continuara: sumar variables de manejo que esta base no tiene (fertilización, edad del cultivo, fecha de recolección, tiempos de fermentación) y lotes que no pasaron por el filtro de certificación. Con los datos actuales, este es el techo del análisis.

## Estructura del repositorio

```
.
├── config/
│   └── config.yaml            # semilla (42), rutas, columnas, algoritmos, modelo elegido,
│                              #   parámetros de estabilidad y nombres de los grupos
├── data/
│   ├── raw/                   # datos originales de CQI, sin modificar
│   ├── interim/               # sin uso en esta iteración
│   ├── processed/             # clustering_input.csv, coffee_clean.csv, clustering_scaler.json,
│   │                          #   coffee_clustered.csv  (generados por 03 y 04)
│   └── external/              # sin uso
├── models/                    # clustering_model.joblib + clustering_metadata.json (fase 04),
│                              #   clustering_pca.joblib + clustering_evaluation.json (fase 05),
│                              #   interpretacion_metadata.json (fase 06)
├── notebooks/
│   ├── 01_comprension_del_negocio.ipynb
│   ├── 02_comprension_de_los_datos.ipynb
│   ├── 03_preparacion_de_los_datos.ipynb
│   ├── 04_modelado.ipynb
│   ├── 05_evaluacion.ipynb
│   └── 06_interpretacion_y_resultados.ipynb
├── reports/
│   ├── figures/               # gráficas .png — EDA (fase 02), PCA y clustering (fase 04),
│   │                          #   diagnóstico de la evaluación (fase 05), perfilado (fase 06)
│   ├── tables/                # clustering_comparison.csv (fase 04) + evaluacion_interna.csv,
│   │                          #   estabilidad.csv, contraste_externo.csv (fase 05) +
│   │                          #   las tablas de perfilado y criterios de negocio (fase 06)
│   ├── dashboard/             # coffee_dashboard_data.csv, dataset plano para Power BI / Tableau
│   └── presentacion/          # diapositivas de la sustentación, ver su propio README
├── app_streamlit/             # dashboard interactivo de exploración, ver su propio README
├── src/
│   ├── config.py              # carga config/config.yaml y resuelve rutas relativas al repo
│   ├── utils.py               # utilidades genéricas (p. ej. resumen de nulos)
│   ├── validation_scripts.py  # valida que el CSV del dashboard esté listo para Power BI
│   ├── pipeline.py            # fachada: prepare_data (03), train_clustering (04),
│   │                          #   evaluate_clustering (05), interpret_clusters (06)
│   ├── data/
│   │   ├── load_data.py       # carga de los CSV crudos de CQI
│   │   └── clean_data.py      # limpieza: registro corrupto, columnas, altitud, Moisture, texto
│   ├── features/
│   │   └── build_features.py  # imputación de perfilado y matriz de clustering estandarizada
│   ├── models/
│   │   ├── train_model.py     # representaciones, barrido rep × algoritmo × k, modelo final
│   │   ├── evaluate_model.py  # validación interna, estabilidad (ARI) y contraste externo
│   │   ├── interpret_model.py # centros en puntos SCA, perfilado por grupo y dataset del tablero
│   │   └── predict_model.py   # asigna lotes nuevos a un grupo (scaler -> PCA -> k-means)
│   └── visualization/
│       └── plots.py           # gráficas reutilizables de EDA y de perfilado por grupo
├── README.md
├── requirements.txt
└── .gitignore
```

**Arquitectura:** capas `data → features → models → visualization`, módulos de
funciones planas (sin `base.py` ni ABCs), orquestadas por `src/pipeline.py`.
Cada fase con código tiene una función que la reproduce entera en una sola
llamada: `prepare_data` (03), `train_clustering` (04), `evaluate_clustering`
(05), `interpret_clusters` (06) — mismos artefactos que los notebooks, salvo
las figuras, que se generan solo ahí.

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
# ... 02, 03, 04, 05, 06

# También, de punta a punta y sin abrir Jupyter:
# jupyter nbconvert --to notebook --execute --inplace notebooks/0{2,3,4,5,6}_*.ipynb
```

Las fases 03 a 06 también corren desde código, sin abrir un notebook:

```python
from src.config import load_config
from src.pipeline import (prepare_data, train_clustering,
                          evaluate_clustering, interpret_clusters)

config = load_config()
prepare_data(config)         # -> data/processed/
train_clustering(config)     # -> models/, data/processed/coffee_clustered.csv, reports/tables/
evaluate_clustering(config)  # -> models/clustering_evaluation.json, clustering_pca.joblib, reports/tables/
interpret_clusters(config)   # -> reports/tables/perfilado_*.csv, reports/dashboard/, models/
```

Y para asignar lotes nuevos a un grupo, sin reentrenar nada:

```python
from src.models.predict_model import assign_new_lotes

# df_nuevos: un DataFrame con los 7 atributos de catación en puntos SCA
assign_new_lotes(df_nuevos, config)   # el mismo DataFrame, con la columna 'grupo'
```

## Autoras

Mariana Valle Moreno
Danna Alejandra Sanchez Monsalve
