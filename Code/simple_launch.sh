#!/bin/bash

LOGFILE="monitor.log"
PYLOGFILE="output/logs/OMAMI_stratified_2_classes.log"

# Crear directorio de logs si no existe
mkdir -p output/logs

# Iniciar monitoreo de CPU y RAM con timestamp (en segundo plano)
echo "Iniciando monitoreo de CPU y RAM con vmstat..."
(while true; do date "+%Y-%m-%d %H:%M:%S" >> "$LOGFILE"; vmstat 1 2 >> "$LOGFILE"; sleep 1; done) &
MONITOR_PID=$!

# Función para matar el monitoreo al salir
trap "echo 'Deteniendo monitoreo'; kill $MONITOR_PID" EXIT

# Ejecutar script Python
echo "Ejecutando OMAMI..." | tee -a "$PYLOGFILE"

python -u scripts/train_multioutput_optuna.py \
  --algorithm 'OMAMI' \
  --optuna false \
  --discard_classes true \
  --filter_EGM false \
  --split_mode 'stratified' \
  --oversampling true \
  >> "$PYLOGFILE" 2>&1

# Verificar si el sistema mató el proceso
echo -e "\n=== Revisión de eventos del sistema por OOM (Out Of Memory): ===" | tee -a "$PYLOGFILE"
dmesg | grep -i -E 'killed process|out of memory' | tail -n 10 | tee -a "$PYLOGFILE"
