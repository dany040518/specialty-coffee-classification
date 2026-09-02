"""Entrenamiento del modelo de clustering (fase 04 - modelado).

TODO fase 04. Aquí irá la lógica para:
- ajustar los algoritmos de `config/config.yaml -> clustering.algorithms`
  (k-means, aglomerativo, gaussian mixture) sobre `data/processed/clustering_input.csv`,
- barrer el número de grupos k en `clustering.k_min`..`clustering.k_max` y
  elegirlo con el coeficiente de silueta,
- opcionalmente comparar con una versión reducida por PCA,
- guardar el modelo elegido y las etiquetas de grupo en `models/`.

El contenido supervisado anterior (regresión logística + SMOTE) se eliminó
al cambiar el proyecto a clustering no supervisado.
"""
