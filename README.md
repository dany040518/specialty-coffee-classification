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
| 5. Evaluación | [`05_evaluacion.ipynb`](notebooks/05_evaluacion.ipynb) | Completo — verificación de reproducibilidad, validación interna en los dos espacios (`pca_2` y los 7 atributos), estabilidad ante 20 semillas y 50 submuestras del 80 % (ARI), estabilidad de la asignación lote a lote, contraste externo contra `Total.Cup.Points >= 80`, revisión de por qué k = 3 y no k = 2, y contraste con los criterios de éxito de la fase 1. |
| 6. Interpretación y resultados | [`06_interpretacion_y_resultados.ipynb`](notebooks/06_interpretacion_y_resultados.ipynb) | Completo — centros de los grupos en puntos SCA, perfilado por país, variedad, procesamiento, color, altitud, humedad, cosecha y defectos, descripción en lenguaje llano de los tres grupos, revisión de los criterios de negocio y export del dataset plano a `reports/dashboard/`. |

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
- **Modelo final:** `pca_2` + **k-means** con **k = 3** grupos (`config -> clustering.final`). Se elige k-means sobre el aglomerativo (resultado casi idéntico, más simple y reproducible) y k = 3 sobre k = 2.
- **Sobre k = 2 (corregido en la fase 5):** la fase 4 había justificado el descarte de k = 2 diciendo que solo reproduce lo que ya dice el umbral de 80 puntos. La fase 5 lo midió y no es así: el ARI de k = 2 contra la binaria de 80 puntos es 0.155, incluso más bajo que el de k = 3 (0.177). Ninguna de las dos particiones reproduce la convención. Lo que sostiene la decisión es que k = 3 aísla en un grupo a 154 de los 180 lotes que no llegan a 80 puntos y deja además dos tramos distinguibles por encima del umbral.
- **PCA guardado (fase 5):** la fase 4 guardó el k-means pero no el PCA sobre el que se ajustó, así que el modelo serializado no servía para lotes nuevos. La fase 5 lo reconstruye y lo guarda en `models/clustering_pca.joblib`.

## El modelo (fase 04)

Modelo final: PCA a 2 dimensiones + k-means con **k = 3**, sobre los 1310 lotes de
arabica, con semilla 42. Métricas internas (`models/clustering_metadata.json`):

| Métrica | Valor | Lectura |
|---|---|---|
| Coeficiente de silueta | 0.428 | Separación moderada. El espacio sensorial está dominado por un solo eje (calidad general), así que los grupos son tres tramos de un continuo más que nubes aisladas. |
| Davies-Bouldin | 0.761 | Más bajo es mejor. |
| Calinski-Harabasz | 1506 | Más alto es mejor. |

Los tamaños de los grupos son 708, 295 y 307 lotes. Qué contiene cada uno se
describe más abajo, en los resultados de la fase 06; el barrido completo de
representación × algoritmo × k está en
[`reports/tables/clustering_comparison.csv`](reports/tables/clustering_comparison.csv).

## Resultados de la evaluación (fase 05)

Los grupos se validaron por tres vías: métricas internas, estabilidad y una
referencia externa que no entró al agrupamiento
(`models/clustering_evaluation.json`, `reports/tables/`).

| Prueba | Resultado | Lectura |
|---|---|---|
| Reproducibilidad | ARI = **1.0** contra las etiquetas de la fase 04 | El modelo se reconstruye desde cero y da exactamente la misma partición. |
| Silueta en `pca_2` | **0.428** | Separación moderada, el mismo número que reportó la fase 04. |
| Silueta en los 7 atributos | **0.296** | En el espacio donde los grupos se interpretan la separación baja casi a la mitad: reducir dimensiones sube la silueta por construcción. Davies-Bouldin pasa de 0.761 a 1.088 y Calinski-Harabasz de 1506 a 812. |
| Estabilidad ante la semilla | ARI medio **0.996** (20 semillas) | La partición no depende del arranque aleatorio; 13 de 20 corridas dan una partición idéntica. |
| Estabilidad ante submuestras | ARI medio **0.952** (50 submuestras del 80 %) | Los grupos sobreviven a quitar un quinto de los lotes; el peor caso da 0.777. |
| Estabilidad de la asignación | **98.4 %** de los lotes conserva su grupo | La variabilidad se concentra en unos 32 lotes de frontera, no está repartida. |
| Contraste externo | ARI **0.177** contra `Total.Cup.Points >= 80` | Los grupos se ordenan por puntaje (78.9, 82.4 y 84.8 de promedio) sin haberlo visto, pero no reproducen el corte de la industria: el umbral parte al grupo 1 casi por la mitad. |

Contra los criterios de éxito de la fase 1 (sección 5.10 del notebook): el criterio
de minería se cumple en estabilidad e interpretabilidad y **se cumple a medias** en
diferenciación, porque una silueta de 0.43 es separación moderada y no excelente.
El criterio de negocio (qué condiciones de origen y manejo se asocian con los
grupos de mejor calidad, y el dashboard) todavía no se puede evaluar: depende de la
fase 06.

## Los tres grupos de cafés (fase 06)

| Grupo | Nombre | n | Puntaje medio | % ≥ 80 | Altitud media | Origen principal | Rasgo que lo distingue |
|---|---|---|---|---|---|---|---|
| 0 | Cafés de puntaje medio, lavados y latinoamericanos | 708 | 82.4 | 96.6 % | 1344 m | Colombia 18 %, México 17 %, Guatemala 14 % | Es el grueso de la base; llega tan limpio como el grupo 2 pero puntúa medio punto menos en todos los atributos |
| 1 | Cafés de puntaje más bajo, de fincas bajas y con más defectos | 295 | 78.9 | 47.8 % | 1230 m | México 32 %, Guatemala 16 % | El único donde la mayoría no llega a 80 puntos, y el único con más defectos: 5.71 de categoría dos por lote contra 3.09 y 2.72 |
| 2 | Cafés de puntaje alto, de fincas altas y orígenes variados | 307 | 84.8 | 99.3 % | 1478 m | Colombia 16 %, Etiopía 12 %, Guatemala 12 % | El más internacional y el de fincas más altas; menos lavado (52 %) y más natural (24 %) que los demás |

Los tres grupos quedan ordenados igual en los siete atributos de catación, sin
una sola excepción, lo que confirma que se separan por nivel general de la taza y
no por un perfil de sabor distinto. Detalle completo en
[`06_interpretacion_y_resultados.ipynb`](notebooks/06_interpretacion_y_resultados.ipynb),
sección 6.9, y en las tablas de `reports/tables/perfilado_*.csv`.

## Dashboard

El dashboard (Power BI o Tableau) se construye por fuera de este repositorio y se
alimenta de [`reports/dashboard/coffee_dashboard_data.csv`](reports/dashboard/coffee_dashboard_data.csv),
que genera la fase 06: **1310 filas** (un lote por fila) × 27 columnas, con las
variables de origen y manejo, el grupo de cada lote, su nombre descriptivo
(`grupo_nombre`), la marca de especialidad (`especialidad`) y el tramo de puntaje
(`rango_puntaje`).

Las vistas sugeridas en la sección 6.12 del notebook 06: un mapa por país con el
tamaño según la cantidad de lotes y el color según el grupo predominante; un
gráfico de altitud contra puntaje coloreado por grupo, que muestra a la vez la
tendencia y el solape entre grupos; barras apiladas con la composición de cada
grupo por procesamiento y variedad, dejando `Desconocido` a la vista; y una
tarjeta por grupo con su nombre, cantidad de lotes, puntaje promedio y porcentaje
de especialidad.

> Cuando el tablero esté publicado va acá el enlace, con una captura y la descripción de sus vistas.

## Limitaciones y consideraciones

Lo primero que hay que saber al leer cualquier resultado de este proyecto es de
dónde salen los datos. La base del CQI reúne cafés que fueron enviados a
certificar, no una muestra del café que se produce en el mundo, y eso se nota en
que el 86.3 % de los lotes ya superaba los 80 puntos antes de que empezáramos.
Por eso el grupo de menor puntaje no es "café malo": es el tramo más bajo de un
conjunto que de entrada ya estaba por encima del promedio, y casi la mitad de sus
lotes igual califica como especialidad. Todo esto vale además solo para arabica,
porque robusta quedó fuera desde la fase 03 por usar otro formulario de catación
y aportar apenas 28 registros.

La segunda consideración es sobre qué tipo de afirmación permiten estos números.
Que el grupo de mejor puntaje tenga más Colombia, más Etiopía y 250 metros más de
altura promedio es una diferencia medible, pero no demuestra que sembrar allí o a
esa altura produzca mejor café. Lo que hay son lotes reales con su suelo, su
clima, su variedad, su manejo poscosecha y el catador que les tocó, todo junto y
sin separar; nadie sembró el mismo café a dos alturas distintas para comparar.
Los rangos de altitud, de hecho, se solapan casi por completo entre los tres
grupos, así que de acá sale un mapa de qué se parece a qué y no una regla del
tipo "siembre por encima de tal altura".

La tercera tiene que ver con la forma de los grupos. Los siete atributos de
catación están tan correlacionados que casi toda la variación entre lotes cabe en
un solo eje, el de calidad general, y por eso los grupos son tres tramos de un
continuo y no nubes separadas. Eso explica que la silueta sea moderada (0.43, y
0.30 en el espacio de los siete atributos) y que las fronteras entre grupos las
haya puesto el algoritmo: unos 32 lotes cambian de grupo según la muestra con la
que se corra. Los grupos son estables (ARI de 0.996 al cambiar la semilla y 0.952
con submuestras del 80 %), que es distinto de decir que tres sea el número
verdadero de perfiles de café que existen. Es una descripción repetible con estos
datos y este algoritmo, y conviene presentarla como niveles de calidad y no como
categorías cerradas.

Dos precisiones más que conviene tener a mano. Los grupos no equivalen a la
categoría de especialidad: el corte de 80 puntos parte al grupo 1 casi por la
mitad y no coincide con ninguna frontera del modelo, con un ARI de solo 0.18
entre las dos particiones. Y hay países y variedades con muy pocos lotes, así que
cualquier lectura sobre ellos, sobre todo en el tablero, tiene que mostrar el `n`
al lado del porcentaje.

## Trabajo pendiente

- Construir el tablero en Power BI o Tableau a partir de `reports/dashboard/coffee_dashboard_data.csv`, y enlazarlo desde este README. Es la única exigencia de la fase 01 que sigue abierta.
- Preparar la sustentación con las descripciones de la sección 6.9 y la tabla de criterios de 6.10.
- Si el proyecto continuara: traer variables de manejo que esta base no tiene (fertilización, edad del cultivo, fecha de recolección, tiempos de fermentación) y lotes que no hayan pasado por el filtro de la certificación. Con los datos actuales, este es el techo del análisis.

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
│   └── dashboard/             # coffee_dashboard_data.csv, el dataset plano para Power BI / Tableau
├── src/
│   ├── config.py             # carga config/config.yaml y resuelve rutas relativas al repo
│   ├── utils.py              # utilidades genéricas (p. ej. resumen de nulos)
│   ├── pipeline.py           # fachada: prepare_data (03), train_clustering (04),
│   │                         #   evaluate_clustering (05), interpret_clusters (06)
│   ├── data/
│   │   ├── load_data.py      # carga de los CSV crudos de CQI
│   │   └── clean_data.py     # limpieza: registro corrupto, columnas, altitud, Moisture, texto
│   ├── features/
│   │   └── build_features.py # imputación de perfilado y matriz de clustering estandarizada
│   ├── models/
│   │   ├── train_model.py    # representaciones, barrido rep × algoritmo × k, modelo final
│   │   ├── evaluate_model.py # validación interna, estabilidad (ARI) y contraste externo
│   │   ├── interpret_model.py # centros en puntos SCA, perfilado por grupo y dataset del tablero
│   │   └── predict_model.py  # asigna lotes nuevos a un grupo (scaler -> PCA -> k-means)
│   └── visualization/
│       └── plots.py          # gráficas reutilizables de EDA y de perfilado por grupo
├── README.md
├── requirements.txt
└── .gitignore
```

**Arquitectura:** capas `data → features → models → visualization` como módulos de
funciones planas (sin `base.py` ni ABCs), orquestadas por `src/pipeline.py`.
Cada fase con código tiene una función que la reproduce entera en una sola
llamada: `prepare_data` (03), `train_clustering` (04), `evaluate_clustering` (05)
e `interpret_clusters` (06). Los artefactos que generan son idénticos a los de
los notebooks; las figuras se generan solo en los notebooks.

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
jupyter notebook notebooks/05_evaluacion.ipynb
jupyter notebook notebooks/06_interpretacion_y_resultados.ipynb

# También, de punta a punta y sin abrir Jupyter:
# jupyter nbconvert --to notebook --execute --inplace notebooks/0{2,3,4,5,6}_*.ipynb
```

Las fases 03 a 06 también se pueden correr desde código, sin abrir un notebook:

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

Y para asignar lotes nuevos a un grupo, sin volver a entrenar nada:

```python
from src.models.predict_model import assign_new_lotes

# df_nuevos: un DataFrame con los 7 atributos de catación en puntos SCA
assign_new_lotes(df_nuevos, config)   # devuelve el mismo DataFrame con la columna 'grupo'
```

## Autoras

Mariana Valle Moreno
Danna Alejandra Sanchez Monsalve
