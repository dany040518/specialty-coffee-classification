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

- **Tema**: fondo crema, tarjetas blancas con sombra suave y tipografía Plus
  Jakarta Sans (Google Fonts). Los colores base están en `.streamlit/config.toml`
  y el resto en la hoja de estilos que inyecta `aplicar_estilos()`. Los tres
  colores de grupo (azul, coral y verde azulado) se eligieron deliberadamente
  fuera de la paleta café del resto de la interfaz, porque tres tonos de marrón
  entre sí son casi indistinguibles en un gráfico; se validaron con el
  validador de paletas del proyecto para separación perceptual, las tres formas
  de daltonismo y contraste contra el fondo.
- **Variables CSS en `style=""`**: Streamlit sanitiza el HTML que inyecta
  `st.markdown` y borra cualquier propiedad `--nombre` de un atributo `style`
  (aunque conserva las propiedades estándar del mismo atributo). El color de
  cada tarjeta de grupo se fija con `border-top-color` y `background` directos,
  no con `var(--tono)`, por esa razón.
- **Círculo de correlaciones**: carga `models/clustering_pca.joblib` (el PCA de
  2 componentes de la fase 05) y calcula `corr(atributo, dimensión) = autovector
  × raíz del autovalor`, la misma fórmula del notebook 04. No depende de los
  filtros de la barra lateral, porque el PCA no se reajusta con cada selección.
  Como varios atributos quedan a menos de 0.15 uno de otro en Dim 2, la etiqueta
  de texto se omite en los que no tienen separación real (se identifican con el
  cursor); la nota debajo del gráfico dice cuáles, calculada en cada corrida.
- **Mapa**: se dibuja con `plotly.graph_objects.Scattergeo`, que traza costas y
  fronteras de forma vectorial. No pide tiles a ningún servidor, así que funciona
  sin conexión y sin API key.
- **Altitud**: 278 lotes no la reportan y la fase 03 decidió no imputarla. El
  filtro los incluye por defecto y hay una casilla para excluirlos; sin ella, ese
  21 % de la base desaparecería sin aviso.
