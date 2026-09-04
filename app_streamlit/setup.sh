#!/bin/bash
# Crea el entorno virtual del dashboard e instala sus dependencias.
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

echo "Instalando dependencias..."
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet -r requirements.txt

DATOS="../reports/dashboard/coffee_dashboard_data.csv"
if [ ! -f "$DATOS" ]; then
    echo
    echo "Advertencia: no se encuentra $DATOS."
    echo "Se genera con la fase 06 del proyecto (notebook 06 o pipeline.interpret_clusters)."
fi

echo
echo "Listo. Para levantar el dashboard:  bash run.sh"
