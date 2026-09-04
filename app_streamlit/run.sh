#!/bin/bash
# Levanta el dashboard. Requiere haber corrido antes setup.sh una vez.
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo "Falta el entorno virtual. Ejecutá primero:  bash setup.sh"
    exit 1
fi

echo "Dashboard en http://localhost:8501  (Ctrl+C para detener)"
exec ./venv/bin/streamlit run app.py
