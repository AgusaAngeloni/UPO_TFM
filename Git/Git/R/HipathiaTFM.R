library(recount3) 
library(edgeR)
library(SummarizedExperiment)
library(hipathia) 
library(ggplot2)
library(ggrepel)
library(dplyr)

# ========== CONFIGURACIÓN: Elige el tejido a comparar

TEJIDO_COMPARAR <- "THYROID"  # Cambiar aquí: "THYROID", "LIVER", "BRAIN", etc.

paths <- c("hsa04660", "hsa04662", "hsa04918", "hsa04919")

# ========== 1. CARGAR DATOS DE DATOS ======================================

# Función para cargar tejido
cargar_tejido <- function(proyecto) {
    gtex_data <- subset(available_projects(), file_source == "gtex" & project == proyecto)
    rse <- create_rse(gtex_data)
    rse <- rse[, rse$gtex.smafrze == "RNASEQ"]
    return(rse)
}

# Cargar Blood (siempre necesario)
rseBLOOD <- cargar_tejido("BLOOD")
rseBLOOD <- rseBLOOD[, rseBLOOD$gtex.smtsd == "Whole Blood"]

# Cargar el tejido elegido
rseTEJIDO <- cargar_tejido(TEJIDO_COMPARAR)

cat("DATOS CARGADOS EXITOSAMENTE\n")

cat("Muestras BLOOD:", length(colnames(rseBLOOD)), "\n")
cat("Muestras", TEJIDO_COMPARAR, ":", length(colnames(rseTEJIDO)), "\n")

# ========== 2. IDENTIFICAR DONANTES COMUNES ===============================

extraer_samples <- function(samples) {
    return(sub("^(([^-]+-[^-]+)).*$", "\\1", samples))
}

samples_BLOOD <- extraer_samples(colnames(rseBLOOD))
samples_TEJIDO <- extraer_samples(colnames(rseTEJIDO))
commons <- intersect(samples_BLOOD, samples_TEJIDO)

cat("BUSCANDO SAMPLES PAREADOS\n")
cat("Samples pareados encontrados:", length(commons), "\n")
cat("\n")

# ========== 3. FILTRAR MUESTRAS COMUNES (solo .1) ===========================

filtrar_muestras <- function(rse, samples_comunes) {
    samples <- colnames(rse)
    donors <- extraer_samples(samples)
    samples_filtradas <- samples[donors %in% samples_comunes & grepl("\\.1$", samples)]
    return(rse[, colnames(rse) %in% samples_filtradas])
}

rseBLOOD_common <- filtrar_muestras(rseBLOOD, commons)
rseTEJIDO_common <- filtrar_muestras(rseTEJIDO, commons)

blood_samples <- colnames(rseBLOOD_common)
tejido_samples <- colnames(rseTEJIDO_common)

cat("Muestras BLOOD:", length(blood_samples), "\n")
cat("Muestras", TEJIDO_COMPARAR, ":", length(tejido_samples), "\n")

# ========== 4. COMBINAR DATASETS ================================================

common_genes <- intersect(rownames(rseTEJIDO_common), rownames(rseBLOOD_common))
rseTEJIDO_common <- rseTEJIDO_common[common_genes, ]
rseBLOOD_common <- rseBLOOD_common[common_genes, ]
rse <- cbind(rseTEJIDO_common, rseBLOOD_common)

# ========== 5. PROCESAMIENTO DE DATOS ===========================================
cat("\n")
cat("PROCESANDO DATOS\n")

# Extraer counts y metadata
counts <- assay(rse, "raw_counts")
metadata <- colData(rse)

# Filtrar genes con baja expresión
keep <- rowSums(counts > 0) >= 5
counts <- counts[keep, ]

# Normalización TMM
dge <- DGEList(counts = counts)
dge <- calcNormFactors(dge, method = "TMM")

# CPM y log2 transform
dge_norm <- cpm(dge, normalized.lib.sizes = TRUE, log = FALSE)
dge_log <- log2(dge_norm + 1)

cat("NORMALIZACIÓN TMM TERMINADA EXITOSAMENTE\n")
cat("\n")

# ========== 6. PREPARACIÓN PARA HIPATHIA ==================================

# Traducir a Entrez IDs
data_entrez <- translate_data(dge_log, "hsa")

# Normalización Hipathia [0,1]
data_entrez_norm <- normalize_data(data_entrez)

# Cargar pathways
pathways <- load_pathways("hsa")
genes_hip <- pathways$all.genes

# Filtrar por genes de hipathia
BLOOD_RNA <- data_entrez_norm[, blood_samples]
BLOOD_RNA <- BLOOD_RNA[rownames(BLOOD_RNA) %in% genes_hip, ]

cat("OBTENCIÓN DE GENES UTILIZADOS EN HIPATHIA:\n")
cat("\n")
cat("Genes en BLOOD_RNA:", nrow(BLOOD_RNA), "\n")
cat("Muestras en BLOOD_RNA:", ncol(BLOOD_RNA), "\n")

# Guardar
write.csv(t(BLOOD_RNA), file = "Data/BLOOD_RNA.csv", row.names = TRUE)

cat("Datos de RNAseq de BLOOD guardados exitosamente.\n")

# ========== 7. ANÁLISIS HIPATHIA

# Crear objeto SummarizedExperiment

data_info <- SummarizedExperiment(
    assays = list(counts = data_entrez_norm),
    colData = DataFrame(
        tissue = colData(rse)$gtex.smts,
        subtissue = colData(rse)$gtex.smtsd
    )
)

# Ejecutar Hipathia
cat("\n")
cat("EJECUTANDO Hipathia ANALISIS \n")
results <- hipathia(data_info, pathways)

# Obtener valores de actividad
path_vals <- get_paths_data(results, matrix = TRUE)

# ========= 8. COMPARACIÓN ESTADÍSTICA =================================

data_design <- as.matrix(colData(data_info))
sample_group_sub <- data_design[colnames(path_vals),"subtissue"]
sample_group_tissue <-  data_design[colnames(path_vals),"tissue"]

comp <- do_wilcoxon(path_vals, sample_group_tissue, 
                  g1 = unique(sample_group_tissue[sample_group_tissue != "Blood"]), 
		  g2= "Blood")

pathways_summary <- get_pathways_summary(comp, pathways)
write.csv(pathways_summary, file = "Data/summary.csv", row.names = TRUE)
write.csv(comp, file = "Data/wilcoxonResult.csv", row.names = TRUE)
cat("Wilcoxon calculado exitosamente")
cat("\n")
# ========== 9. VISUALIZACIÓN PCA ======================================

ranked_path_vals <- path_vals[order(comp$p.value, decreasing = FALSE),]
pca_model <- do_pca(ranked_path_vals[1:ncol(ranked_path_vals),])
var_explained <- (pca_model$sdev^2) / sum(pca_model$sdev^2)
cat("\n")
cat("Valores de PC1 y PC2: ", round(var_explained[1:2] * 100, 1),"\n")
cat("\n")

pca_plot(
  pca_model,
  sample_group_tissue,
  legend = TRUE,
  main = "PCA Blood vs Thyroid"
)

mtext(
  paste0("PC1 (", round(var_explained[1] * 100, 1), "%)"),
  side = 1, line = 3
)

mtext(
  paste0("PC2 (", round(var_explained[2] * 100, 1), "%)"),
  side = 2, line = 3
)

# Guardar PCA
svg(paste0("Graph/pca_blood_vs_", tolower(TEJIDO_COMPARAR), ".svg"), 
    width = 12, height = 10)
pca_plot(pca_model, sample_group_tissue, legend = TRUE,
         main = paste("BLOOD vs", TEJIDO_COMPARAR))
dev.off()
png(paste0("Graph/pca_blood_vs_", tolower(TEJIDO_COMPARAR), ".png"), 
    width = 800, height = 600, res = 100)
cat("PCA guardado exitosamente")
	 
colors_de <- node_color_per_de(
  results,
  pathways,
  sample_group_tissue,
  "Thyroid",
  "Blood",
  colors = c(
    down = "blue",
    neutral = "#F7F7F7",
    up = "red"
  )
)

for (p in paths) {

  png(
    filename = paste0("Graph/comparison_", p, ".png"),
    width  = 3200,  
    height = 2200,
    res    = 300
  )

  par(
    mar = c(1, 3, 2, 4),
    oma = c(0, 0, 0, 0),
    xaxs = "i",
    yaxs = "i",
    cex      = 1.8,   
    cex.axis = 1.4,
    cex.main = 1.1,   
    font     = 2       
  )

  pathway_comparison_plot(
    comp,
    metaginfo = pathways,
    pathway = p,
    node_colors = colors_de
  )

  dev.off()
}

cat("Gráficos de rutas guardados exitosamente")

# ========= 10. GUARDAR RESULTADOS FINALES ==================================

hipathia_blood <- t(path_vals[, blood_samples])
hipathia_tejido <- t(path_vals[, tejido_samples])

write.csv(hipathia_blood, file = "Data/hipathia_blood.csv", row.names = TRUE)
write.csv(hipathia_tejido,
          file = paste0("Data/hipathia_", tolower(TEJIDO_COMPARAR), ".csv"),
          row.names = TRUE)

cat("\n¡ANÁLSIS COMPLETADO!\n")
cat("Tejido comparado:", TEJIDO_COMPARAR, "\n")

