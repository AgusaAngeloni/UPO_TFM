#!/bin/bash
#SBATCH -J ags_comparacion
#SBATCH -p day
#SBATCH -N 1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32G  
#SBATCH -t 24:00:00    
#SBATCH -o ./output_jobs/slurm.%N.%j.out
#SBATCH -e ./output_jobs/slurm.%N.%j.err 

# CONFIGURACIÓN DEL USUARIO (EDITAR SOLO ESTA SECCIÓN)

# Ruta base del proyecto
PROJECT_DIR="/home/aangelo/Hipathia/Final/Git"

# Ruta al entorno Conda
CONDA_ENV="/home/aangelo/.conda/envs/UPO_TFM"

# =====================================================
# NO MODIFICAR A PARTIR DE AQUÍ

# Ir al directorio base del proyecto
cd "$PROJECT_DIR" || { echo "ERROR: No se pudo acceder a $PROJECT_DIR"; exit 1; }

# Configurar entorno
export R_HOME="$CONDA_ENV/lib/R"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_ENV/bin:$PATH"

mkdir -p $PROJECT_DIR/Data
mkdir -p $PROJECT_DIR/Graph

# Ir al subdirectorio de trabajo
cd "$PROJECT_DIR" || { echo "ERROR: No se pudo acceder a Final"; exit 1; }

python Python/comparacionTFM.py --n_jobs=$SLURM_CPUS_PER_TASK


