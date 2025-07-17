BiocManager::install("hipathia")
BiocManager::install("recount3")
BiocManager::install("SummarizedExperiment")
BiocManager::install("GenomicRanges")
BiocManager::install("BiocGenerics")
library(recount3) 
library(biomaRt)
library(edgeR)
library(SummarizedExperiment)
library(hipathia) 

### 1 - OBTENCIÓN DE LOS RNA-SEQ DE GTEx
## Mediante recount3, primero selecciono los proyectos (tejidos) por su
## nombre en available_projects() y mediante create_rse() obtengo
## RangesSummarizedExperiment. 
## Luego juntos los ditintos tejidos que mq interesan estudiar

####--- BLOOD
gtex_blood <- subset(available_projects(), file_source == "gtex" & 
                       project =="BLOOD")
rseBLOOD <- create_rse(gtex_blood)
####--- OVARY
gtex_ovar <- subset(available_projects(), file_source == "gtex" & 
                      project =="OVARY")
rseOVAR <- create_rse(gtex_ovar)
###--- UTERUS
gtex_uter <- subset(available_projects(), file_source == "gtex" & 
                      project =="UTERUS")
rseUTER <- create_rse(gtex_uter)
###--- THYROIDES
gtex_thyr <- subset(available_projects(), file_source == "gtex" & 
                      project =="THYROID")
rseTHYR<- create_rse(gtex_thyr)

##--- COOMBINADO
rse <- cbind(rseBLOOD, rseTHYR )#seUTER,rseOVAR
## Obtengo por separado un array con los conteso crudos de RNA-seq y los 
## datos clínicos y experimentales en metadata
counts <- assay(rse, "raw_counts")  #Matriz de conteos crudos
metadata <- colData(rse)            #df de datos clínicos y experimentales

### 2 - PRE-FILTRADOO
keep <- rowSums(counts > 0) >= 5  # Genes expresados en ≥5 muestras
counts <- counts[keep, ] #De 63856 genes a 55424 genes

### 3 - NORMALIZACIÓN TMM EN DATOS CRUDOS 

# Normalización TMM = Trimmed Mean of M-values:
# método para corregir diferencias en la profundidad de secuenciación y 
# en la composición global del RNA entre muestras.
# Conversión counts crudos a CPM (Counts Per Million),
# usando tamaños de librería ajustados con factores de normalización TMM 
# (calcula lib.size * norm.factors)
dge <- DGEList(counts = counts) # clase DGEList (Differential Gene Expression List)
dge_norm <- cpm(dge, 
                normalized.lib.sizes = TRUE, 
                log = FALSE)                              
                 
dge_normalized <- log2(dge_norm + 1) # transformación log2 para obtener valores continuos comparables 

### 4 - CAMBIO DE ENSEMBL A ENTREZ (Hipathia)
data_entrez <- translate_data(dge_normalized, "hsa")
  # translated ids = 23736 (0.43) 
  # untranslated ids = 31688 (0.57) 
  # multihit ids = 181 (0.0033) 
## De los 55424 genes por Ensembl, se reconocen 23638 con id Entrez en Hipathia
table(is.na(rownames(data_entrez)))
# IMPORTANTE: que no existan filas con id no reconocidos por Hipathia

### 5 - NORMALIZACION a [0,1] (Hipathia)
data_entrez_norm <- normalize_data(data_entrez)

### 6 - OBJETO FINAL PARA HIPATHIA SummarizedEperiment
data_info <- SummarizedExperiment(
  assays = list(counts = data_entrez_norm), #counts normalizados
  colData = colData(rse)$gtex.smts #metadatos tissuetype 
)

### 7 - DESCARGO PATHWAYS HUMANOS (Hipathia)
pathways <- load_pathways("hsa")
    # Loaded 146 pathways

### 8 - CALCULO DE ACTIVIDAD DE RUTAS
## HIPATHIA - result
results <- hipathia(data_info, pathways) # utiliza las subrutas efectoras
                                         # opción decompose=TRUE utiliza las subrutas descompuestas

## Obtengo los valores de actividad de la señal como matriz
path_vals <- get_paths_data(results, matrix = TRUE)
## Obtengo los valores de actividad de la señal como SummarizedExperiment
#path <- get_paths_data(results)
head(path_vals)  

# Datos de tipo de tejido
data_design <- as.matrix(colData(data_info))
sample_group <- data_design[colnames(path_vals),"X"]

### -------- Comparación de las características entre diferentes grupos de muestras
comp <- do_wilcoxon(path_vals, sample_group, g1 = "Blood", g2 = "Thyroid")
hhead(comp)
        # UP/DOWN  statistic       p.value    FDRp.value
        #P-hsa03320-37    DOWN  -9.297917  1.184145e-20  1.577739e-20
        #P-hsa03320-61    DOWN -21.828683 1.232482e-105 6.494762e-105
        #P-hsa03320-46      UP   4.746509  2.059869e-06  2.343429e-06
        #P-hsa03320-57    DOWN  -1.203628  2.243889e-01  2.325711e-01
        #P-hsa03320-64    DOWN  -1.324208  1.371008e-01  1.428102e-01

pathways_summary <- get_pathways_summary(comp, pathways)

##### PCA
ranked_path_vals <- path_vals[order(comp$p.value, decreasing = FALSE),]
pca_model <- do_pca(ranked_path_vals[1:ncol(ranked_path_vals),])
pca_plot(pca_model, sample_group, legend = T, main="Blood vs Thyroid")

##### Pathway comparison
colors_de <- node_color_per_de(results, pathways, sample_group,"Blood","Thyroid", 
                                colors = "hipathia")
pathway_comparison_plot(comp, metaginfo = pathways, pathway = "hsa04110",
                        node_colors = colors_de)

