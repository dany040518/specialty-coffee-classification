"""Carga de la configuración central del proyecto (config/config.yaml).

"""

from pathlib import Path
from typing import Any

import yaml

# Raíz del repositorio: dos niveles arriba de este archivo (src/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Carga config/config.yaml y devuelve su contenido como diccionario.

    Args:
        config_path: ruta al archivo YAML de configuración.

    Returns:
        Diccionario con la configuración del proyecto (paths, semilla, target, models).
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Convierte una ruta relativa (tal como aparece en config.yaml) en una ruta
    absoluta anclada a la raíz del repositorio, para que los notebooks nunca
    dependan del directorio de trabajo desde el que se ejecutan.
    """
    return REPO_ROOT / relative_path
