# Dashboard de perfiles de calidad de café

Interfaz interactiva sobre los tres grupos que produjo la fase 06 del proyecto.
Lee `../reports/dashboard/coffee_dashboard_data.csv` (1310 lotes de arabica) y no
recalcula el modelo: solo lo presenta.

## Ejecutar

```bash
bash setup.sh   # una sola vez: crea venv/ e instala las dependencias
bash run.sh     # levanta el dashboard en http://localhost:8501
```

Si el CSV no existe todavía, se genera desde la raíz del proyecto:

```python
from src.config import load_config
from src.pipeline import interpret_clusters
interpret_clusters(load_config())
```

## Qué contiene

| Sección | Contenido |
|---|---|
| Indicadores | Lotes, porcentaje de especialidad, puntaje medio, altitud media y origen principal de la selección activa |
| 01 Los tres perfiles | Una tarjeta por grupo con su nombre, tamaño, puntaje, altitud, origen, beneficio y defectos |
| 02 Espacio sensorial | Dispersión Flavor vs Aroma y círculo de correlaciones de los siete atributos contra las dos primeras componentes del PCA |
| 03 Distribución | Caja y bigotes e histograma del puntaje total, con el umbral de 80 marcado |
| 04 Origen | Mapa mundial por país y ranking de los orígenes más frecuentes |
| 05 Correlaciones | Matriz entre los siete atributos, el puntaje total y la altitud |
| 06 Contraste estadístico | ANOVA de una vía sobre el puntaje y estadísticas descriptivas por grupo |
| 07 Registros | Tabla con buscador por país o variedad |

Los filtros de la barra lateral (grupo, puntaje, país, especialidad y altitud)
afectan a todas las secciones a la vez.

## Detalles de implementación

- **Tema**: fondo crema, tarjetas blancas y tipografía Plus Jakarta Sans
  (Google Fonts) — colores base en `.streamlit/config.toml`, el resto en la
  hoja de estilos de `aplicar_estilos()`. Los tres colores de grupo (azul,
  coral, verde azulado) se validaron aparte de la paleta café: tres tonos de
  marrón son casi indistinguibles entre sí en un gráfico.
- **Colores de tarjeta vía `border-top-color`/`background`, no `var(--tono)`**:
  Streamlit borra cualquier propiedad `--nombre` de un atributo `style` al
  sanitizar el HTML de `st.markdown`, aunque conserve las propiedades
  estándar del mismo atributo.
- **Círculo de correlaciones**: carga el PCA de 2 componentes de la fase 05
  (`models/clustering_pca.joblib`) y calcula `corr(atributo, dimensión) =
  autovector × raíz del autovalor`, la fórmula del notebook 04. Fijo, no
  depende de los filtros. Varios atributos quedan a menos de 0.15 uno de otro
  en Dim 2; la etiqueta se omite donde no hay separación real (identificables
  con el cursor), y una nota debajo dice cuáles.
- **Mapa**: `plotly.graph_objects.Scattergeo`, vectorial — no pide tiles a
  ningún servidor, funciona sin conexión y sin API key.
- **Altitud**: 278 lotes no la reportan (la fase 03 decidió no imputarla). El
  filtro los incluye por defecto, con una casilla para excluirlos — sin ella,
  ese 21 % desaparecería sin aviso.
