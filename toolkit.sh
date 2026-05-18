#!/usr/bin/env bash

set -e

SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT_PATH" ]; do
    LINK_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
    SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
    case "$SCRIPT_PATH" in
        /*) ;;
        *) SCRIPT_PATH="$LINK_DIR/$SCRIPT_PATH" ;;
    esac
done

ROOT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
VENV_PYTHON="$ROOT_DIR/.toolkit2venv/bin/python"

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ "$1" = "---help" ] || [ "$1" = "help" ]; then
    echo "Uso:"
    echo "  estimator XX archivo_calibracion archivo_evaluacion outdir"
    echo "            [K] [--remain] [--trim:VALOR]"
    echo "  estimator ts archivo_calibracion archivo_evaluacion outdir"
    echo "            [K] [--remain] [ce|dE|DE]"
    echo "  estimator lm|sx archivo_calibracion_agrupada archivo_evaluacion_agrupada outdir"
    echo "            K --grouped [--remain] [--trim:VALOR]"
    echo "  estimator ts archivo_calibracion_agrupada archivo_evaluacion_agrupada outdir"
    echo "            K --grouped [--remain]"
    echo
    echo "Entradas y orden:"
    echo "  XX                  Tipo de estimador: lm, sx o ts"
    echo "  archivo_calibracion Archivo de calibracion"
    echo "  archivo_evaluacion  Archivo de evaluacion"
    echo "  outdir              Directorio raiz donde se guardan los"
    echo "                      resultados. Es obligatorio"
    echo "  K                   Numero de grupos. Si no se indica sin --grouped,"
    echo "                      usa 100. Con --grouped es obligatorio y solo"
    echo "                      etiqueta la carpeta del experimento"
    echo "  --remain            Si se incluye, el resto se guarda como"
    echo "                      bloque separado (t=1). Si no aparece, t=0"
    echo "  --grouped           Los ficheros ya estan agrupados como:"
    echo "                      m media_estimada media_empirica"
    echo "                      En este modo K y t no se usan para calcular,"
    echo "                      solo para hacer comprensible la ruta"
    echo "  --trim:VALOR        Solo para lm y sx. Activa triming con"
    echo "                      ese limite"
    echo "  ce|dE|DE            Solo para ts. Criterio de optimizacion."
    echo "                      Por defecto ce"
    echo
    echo "Notas:"
    echo "  lm  -> ajuste con Levenberg-Marquardt"
    echo "  sx  -> ajuste con Simplex"
    echo "  ts  -> temperature scaling"
    echo
    echo "Metricas de salida:"
    echo "  Ecal -> error calibrado (%)"
    echo "  Ee   -> error estimado de entrada (%), media de 1-Pmax"
    echo "  E    -> error empirico (%)"
    echo "  DE   -> |Ecal - E| (%)"
    echo
    echo "Ejemplos:"
    echo 
    echo "  estimator lm examples/calibration.txt examples/evaluation.txt results 3 \\"
    echo "      --remain"
    echo
    echo "  estimator ts examples/calibration_ts.txt examples/evaluation_ts.txt \\"
    echo "      results 3 dE"
    exit 0
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "No existe .toolkit2venv. Ejecuta ./install.sh primero."
    exit 1
fi

exec "$VENV_PYTHON" "$ROOT_DIR/code/wraper.py" "$@"
