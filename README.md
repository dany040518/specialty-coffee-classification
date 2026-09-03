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
- **Modelo final:** `pca_2` + **k-means** con **k = 3** grupos (`config -> clustering.final`). Se elige k-means sobre el aglomerativo (resultado casi idéntico, más simple y reproducible) y k = 3 sobre k = 2.
- **Sobre k = 2 (corregido en la fase 5):** la fase 4 había justificado el descarte de k = 2 diciendo que solo reproduce lo que ya dice el umbral de 80 puntos. La fase 5 lo midió y no es así: el ARI de k = 2 contra la binaria de 80 puntos es 0.155, incluso más bajo que el de k = 3 (0.177). Ninguna de las dos particiones reproduce la convención. Lo que sostiene la decisión es que k = 3 aísla en un grupo a 154 de los 180 lotes que no llegan a 80 puntos y deja además dos tramos distinguibles por encima del umbral.
- **PCA guardado (fase 5):** la fase 4 guardó el k-means pero no el PCA sobre el que se ajustó, así que el modelo serializado no servía para lotes nuevos. La fase 5 lo reconstruye y lo guarda en `models/clustering_pca.joblib`.

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
├── models/                    # clustering_model.joblib + clustering_metadata.json (fase 04),
│                              #   clustering_pca.joblib + clustering_evaluation.json (fase 05)
├── notebooks/
│   ├── 01_comprension_del_negocio.ipynb
│   ├── 02_comprension_de_los_datos.ipynb
│   ├── 03_preparacion_de_los_datos.ipynb
│   ├── 04_modelado.ipynb
│   ├── 05_evaluacion.ipynb
│   └── 06_interpretacion_y_resultados.ipynb  # pendiente
├── reports/
│   ├── figures/               # gráficas .png — EDA (fase 02), PCA y clustering (fase 04),
│   │                          #   diagnóstico de la evaluación (fase 05)
│   ├── tables/                # clustering_comparison.csv (fase 04) + evaluacion_interna.csv,
│   │                          #   estabilidad.csv, contraste_externo.csv (fase 05)
│   └── dashboard/             # dataset plano para Power BI / Tableau (pendiente, fase 06)
├── src/
│   ├── config.py             # carga config/config.yaml y resuelve rutas relativas al repo
│   ├── utils.py              # utilidades genéricas (p. ej. resumen de nulos)
│   ├── pipeline.py           # fachada: prepare_data (03), train_clustering (04), evaluate_clustering (05)
│   ├── data/
│   │   ├── load_data.py      # carga de los CSV crudos de CQI
│   │   └── clean_data.py     # limpieza: registro corrupto, columnas, altitud, Moisture, texto
│   ├── features/
│   │   └── build_features.py # imputación de perfilado y matriz de clustering estandarizada
│   ├── models/
│   │   ├── train_model.py    # representaciones, barrido rep × algoritmo × k, modelo final
│   │   ├── evaluate_model.py # validación interna, estabilidad (ARI) y contraste externo
│   │   └── predict_model.py  # asignación de lotes nuevos a un grupo (pendiente, fase 06)
│   └── visualization/
│       └── plots.py          # gráficas reutilizables de EDA
├── README.md
├── requirements.txt
└── .gitignore
```

**Arquitectura:** capas `data → features → models → visualization` como módulos de
funciones planas (sin `base.py` ni ABCs), orquestadas por `src/pipeline.py`.
`prepare_data(config)` reproduce toda la fase 03, `train_clustering(config)` toda
la fase 04 y `evaluate_clustering(config)` toda la fase 05, cada una en una sola
llamada y con los mismos artefactos que producen los notebooks (las figuras se
generan solo en los notebooks).

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

# También, de punta a punta y sin abrir Jupyter:
# jupyter nbconvert --to notebook --execute --inplace notebooks/0{2,3,4,5}_*.ipynb
```

Las fases 03, 04 y 05 también se pueden correr desde código:

```python
from src.config import load_config
from src.pipeline import prepare_data, train_clustering, evaluate_clustering

config = load_config()
prepare_data(config)         # -> data/processed/
train_clustering(config)     # -> models/, data/processed/coffee_clustered.csv, reports/tables/
evaluate_clustering(config)  # -> models/clustering_evaluation.json, clustering_pca.joblib, reports/tables/
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

> El perfilado detallado por origen y manejo (fase 06) sigue pendiente. La
> validación de estos grupos está en la sección siguiente.

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

## Dashboard

El dashboard (Power BI o Tableau) se construye por fuera de este repositorio y se
alimentará de un dataset plano (grupo de cada lote + sus características) que
generará la fase 06 en `reports/dashboard/`.

> TODO: una vez publicado, agregar aquí el enlace/captura y describir qué vistas contiene.

## Limitaciones y trabajo futuro

- La base de datos solo incluye cafés que buscaron una certificación de calidad de CQI, así que no representa a todo el café que se produce; el **86.3 %** de los lotes ya supera los 80 puntos. El "grupo de calidad más baja" lo es dentro de esta base, no dentro del café del mundo.
- Se trabajó solo con arabica; robusta quedó fuera por formulario de catación distinto, alta nulidad y tamaño (28 filas).
- La separación entre grupos es moderada (silueta 0.43 en `pca_2` y **0.30** en el espacio de los 7 atributos, donde los grupos tienen que interpretarse) porque los atributos sensoriales están muy correlacionados: casi toda la variación cabe en un eje de calidad general. El agrupamiento se usa como base **descriptiva**, no como una frontera nítida.
- Los grupos no equivalen a la categoría de especialidad: el corte de 80 puntos parte al grupo 1 casi por la mitad y no coincide con ninguna frontera del modelo (fase 5, sección 5.8).
- Los 32 lotes que quedan en las fronteras entre tramos cambian de grupo según la muestra (fase 5, sección 5.7). La partición es estable, esos lotes no.
- Lo que la fase 5 validó son los grupos, no el perfilado: las asociaciones con país, variedad o altitud que salgan en la fase 6 son descriptivas y no admiten lectura causal.
- Hay países y variedades con muy pocos lotes; las conclusiones sobre ellos en la fase 06 deben tomarse con cautela.
- Falta la fase 06 (interpretación, perfilado por origen y manejo, y dashboard).

## Autoras

Mariana Valle Moreno
Danna Alejandra Sanchez Monsalve
