"""Evaluación de los grupos del clustering (fase 05 - evaluación).

TODO fase 05. Aquí irá la lógica para:
- validación interna: coeficiente de silueta, Davies-Bouldin, Calinski-Harabasz
  (`config/config.yaml -> validation.internal_metrics`),
- estabilidad: repetir el agrupamiento con distintas semillas / submuestras,
- contraste externo: cruzar los grupos con `Total.Cup.Points >= 80`
  (`validation.reference_column` / `specialty_threshold`).

El contenido supervisado anterior (accuracy/precision/recall/F1/ROC-AUC,
matriz de confusión) se eliminó al cambiar el proyecto a clustering.
"""
