library(SummarizedExperiment)
library(hipathia) 
library(stringr)
library(tidyverse)

armar_inputs_wilcoxon <- function(blood_df, tissue_df, tissue_label = "Thyroid") {

  # -------------------------
  # Extraer donors
  # -------------------------
  donor_blood <- sub("^(([^-]+-[^-]+)).*$", "\\1", blood_df[[1]])
  donor_tissue <- sub("^(([^-]+-[^-]+)).*$", "\\1", tissue_df[[1]])

  commons <- intersect(donor_blood, donor_tissue)

  # -------------------------
  # Filtrar muestras comunes
  # -------------------------
  blood_common <- blood_df %>%
    mutate(donor = donor_blood) %>%
    filter(donor %in% commons) %>%
    select(-donor)

  tissue_common <- tissue_df %>%
    mutate(donor = donor_tissue) %>%
    filter(donor %in% commons) %>%
    select(-donor)

  # -------------------------
  # Intersección de rutas
  # -------------------------
  rutas_comunes <- intersect(
    colnames(blood_common)[-1],
    colnames(tissue_common)[-1]
  )

  blood_common <- blood_common %>%
    select(1, all_of(rutas_comunes)) %>%
    arrange(.[[1]])

  tissue_common <- tissue_common %>%
    select(1, all_of(rutas_comunes)) %>%
    arrange(.[[1]])

  # -------------------------
  # Construir path_vals
  # -------------------------
  path_vals <- bind_rows(blood_common, tissue_common)

  path_vals_t <- as.matrix(t(path_vals))
  colnames(path_vals_t) <- path_vals_t[1, ]
  path_vals_t <- path_vals_t[-1, ]
  path_vals_t <- apply(path_vals_t, 2, as.numeric)
  rownames(path_vals_t) <- colnames(path_vals)[-1]

  # -------------------------
  # Sample groups
  # -------------------------
  blood_samples <- cbind(
    Sample = blood_common[,1],
    Group = "Blood"
  )

  tissue_samples <- cbind(
    Sample = tissue_common[,1],
    Group = tissue_label
  )

  sample_groups <- rbind(blood_samples, tissue_samples)
  sample_groups <- t(sample_groups)
  colnames(sample_groups) <- sample_groups[1, ]
  sample_groups <- sample_groups[2, , drop = FALSE]

  return(list(
    path_vals_t = path_vals_t,
    sample_groups = sample_groups
  ))
}

# -------------------------
# CONFIGURACIÓN
# -------------------------
out <- "Data"
blood <- read_csv("Data/hipathia_blood.csv")

modelos <- c("RandomForest", "RidgeCV", "ElasticNet")

out_base <- file.path(out, "wilcoxon")
dir.create(out_base, showWarnings = FALSE)

# -------------------------
# LOOP PRINCIPAL
# -------------------------
for (modelo in modelos) {

  message("Procesando modelo: ", modelo)

  pred_path <- file.path(modelo, "predicciones")
  out_modelo <- file.path(out_base, modelo)
  dir.create(out_modelo, showWarnings = FALSE)

  tests <- list.files(pred_path, pattern = "fold[0-9]+_test\\.csv$", full.names = TRUE)

  for (test_file in tests) {

    fold <- str_extract(basename(test_file), "fold[0-9]+")
    message("  → ", fold)

    # -------------------------
    # Cargar TRUE / PRED
    # -------------------------
    df_test <- read_csv(test_file)
    df_pred <- read_csv(
      file.path(pred_path, paste0(fold, "_predicciones.csv"))
    )

    # -------------------------
    # TRUE
    # -------------------------
    inputs_true <- armar_inputs_wilcoxon(
      blood_df  = blood,
      tissue_df = df_test,
      tissue_label = "Thyroid"
    )

    # -------------------------
    # PRED
    # -------------------------
    inputs_pred <- armar_inputs_wilcoxon(
      blood_df  = blood,
      tissue_df = df_pred,
      tissue_label = "Thyroid"
    )

    comp_true <- do_wilcoxon(
      inputs_true$path_vals_t,
      inputs_true$sample_groups,
      g1 = "Thyroid",
      g2 = "Blood"
    )

    comp_pred <- do_wilcoxon(
      inputs_pred$path_vals_t,
      inputs_pred$sample_groups,
      g1 = "Thyroid",
      g2 = "Blood"
    )

    # -------------------------
    # Guardar resultados Wilcoxon
    # -------------------------

    write.csv(
      comp_true,
      file = file.path(
        out_modelo,
        paste0(fold, "_wilcoxon_true.csv")
      ),
      row.names = TRUE
    )

    write.csv(
      comp_pred,
      file = file.path(
        out_modelo,
        paste0(fold, "_wilcoxon_pred.csv")
      ),
      row.names = TRUE
    )

  }
}
