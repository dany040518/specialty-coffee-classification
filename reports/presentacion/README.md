# Presentación de sustentación

Diapositivas del proyecto (16), en HTML.

| Archivo | Qué es |
|---|---|
| `deck.html` | **La presentación lista.** Un solo archivo, con las figuras embebidas. Se abre con doble clic en cualquier navegador. |
| `deck.src.html` | Plantilla editable (las figuras van como marcadores `data-img="..."`). |
| `build.py` | Regenera `deck.html` a partir de `deck.src.html` incrustando las figuras de `../figures/`. |

## Usar

- **Presentar / ver:** abrir `deck.html`. Navegar con `←` `→` o la barra espaciadora; también con clic (mitad derecha avanza, izquierda retrocede). Se ajusta al tamaño de la ventana; pantalla completa con `F11`.
- **Editar el contenido:** modificar `deck.src.html` y luego:

  ```bash
  python reports/presentacion/build.py
  ```

Las figuras son las de `reports/figures/` generadas por los notebooks 02–06.
