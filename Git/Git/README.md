# Universidad Pablo de Olavide
# TFM – Máster Análisis Bioinformático Avanzado
# Estudio asociativo y desarrollo de modelos predictivos de la actividad de las rutas de señalización celular entre tejidos.
## Autora: Angeloni Agustina Renata

Este repositorio contiene el código y los recursos utilizados en el Trabajo Fin de Máster (TFM) 
orientado a evaluar si la expresión génica en sangre periférica puede emplearse para inferir la actividad funcional 
de rutas de señalización en tejidos no sanguíneos, utilizando datos del proyecto GTEx y la herramienta Hipathia.

El flujo de trabajo integra análisis en R y Python, ejecutados en un entorno computacional reproducible y automatizados 
mediante scripts Bash bajo un sistema de colas SLURM.

---

## Descarga del repositorio

El repositorio puede descargarse mediante:

```
git clone <https://github.com/AgusaAngeloni/UPO_TFM.git>
cd <UPO_TFM>
```

---

## Entorno computacional y dependencias

Todo el análisis se realizó utilizando entornos virtuales gestionados con Conda.  
La configuración completa del entorno se encuentra definida en el archivo `environment.yml`.

### Creación del entorno Conda

```
conda env create -f environment.yml
conda activate UPO_TFM
```

### Versiones principales utilizadas

- R 4.3.3
- Python 3.10.18
- scikit-learn 1.7.1
- Hipathia
- edgeR
- recount3

Las librerías adicionales de R utilizadas incluyen, entre otras:
- edgeR
- recount3
- Hipathia
- limma
- tidyverse

---

## Estructura del repositorio

```
├── Data/
├── Graphs/
├── Scripts/
│   ├── R/
│   │   ├── HipathiaTFM.R
│   │   └── WilcoxonAnalysis.R
│   ├── Python/
│   │   ├── RidgeCV.py
│   │   ├── ElasticNet.py
│   │   ├── RandomForest.py
│   │   └── comparacionTFM.py
│   └── Bash/
│       ├── Hipathia.sh
│       ├── Modelos.sh
│       ├── wilcoxon.sh
│       └── comparacionTFM.sh
│
├── environment.yml
└── README.md
```

### Descripción de carpetas

- **Data/**: contiene las matrices de expresión génica, matrices de actividad de rutas y resultados intermedios y finales utilizados en el análisis.
- **Scripts/**: incluye los scripts en R, Python y Bash utilizados para ejecutar el flujo completo del análisis.
- **Graphs/**: almacena los gráficos finales generados y utilizados en la memoria del TFM.
- **environment.yml**: definición del entorno Conda para garantizar la reproducibilidad.

---

## Configuración de rutas (IMPORTANTE)

Antes de ejecutar los scripts `.sh`, es necesario editar la sección de configuración del usuario en cada uno de ellos:

```
# CONFIGURACIÓN DEL USUARIO (EDITAR SOLO ESTA SECCIÓN)
PROJECT_DIR="/ruta/al/proyecto"
CONDA_ENV="/ruta/al/entorno_conda"
```
Estas rutas deben apuntar al directorio base del proyecto y al entorno Conda previamente creado.

---

## Ejecución del flujo de trabajo

El análisis debe ejecutarse en el siguiente orden, utilizando un sistema SLURM:

1. **Hipathia.sh**  
   Cálculo de la actividad de rutas de señalización a partir de los datos transcriptómicos.

2. **Modelos.sh**  
   Entrenamiento y evaluación de los modelos predictivos (RidgeCV, Multitask ElasticNet y Random Forest).

3. **wilcoxon.sh**  
   Análisis estadístico de las diferencias entre valores reales y predichos de actividad de rutas.

4. **comparacionTFM.sh**  
   Comparación global de desempeño entre modelos y generación de gráficos finales.

Ejemplo de ejecución:

```
sbatch Scripts/Bash/Hipathia.sh
```
---

## Gráficos y resultados

La carpeta **Graphs/** contiene únicamente las figuras finales utilizadas en la memoria del TFM.  
Los resultados numéricos completos se encuentran almacenados en la carpeta **Data/**.

---

## Reproducibilidad

Todo el código, las versiones de software y la estructura del flujo de trabajo están diseñados 
para permitir la reproducción completa del análisis en un entorno HPC compatible con SLURM.

