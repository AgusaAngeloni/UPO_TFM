import rpy2.robjects as robjects
import pandas as pd
from rpy2.robjects import pandas2ri

pandas2ri.activate()
robjects.r('''
    library (recount3)
    library(SummarizedExperiment)
    library(hipathia)
    gtexBlood <- subset(available_projects(),
           file_source == "gtex" & project == "BLOOD")
    rseBLOOD <- create_rse(gtexBlood)
    rseBLOOD <- assay(rseBLOOD)
    pathways <- load_pathways("hsa")
''')

# Traer el objeto gtexBlood de R a Python
gtex_df = pandas2ri.rpy2py(robjects.r['gtexBlood'])

# Mostrar las primeras filas en Python (head de pandas)
print(gtex_df.head())