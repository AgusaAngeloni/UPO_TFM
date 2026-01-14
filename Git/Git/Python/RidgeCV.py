import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

print("\n" + "="*50)
print("INICIANDO RIDGE CV GLOBAL")
print("="*50)

pathway_result = "RidgeCV"
os.makedirs(pathway_result, exist_ok=True)

print("Cargando datos...")
Xpd = pd.read_csv("Data/BLOOD_RNA.csv", index_col=0).sort_index(axis=0)
Ypd = pd.read_csv("Data/hipathia_thyroid.csv", index_col=0).sort_index(axis=0)

genes = Xpd.columns.tolist()
vias_inicial = Ypd.columns.tolist()
samples = Xpd.index.tolist()

print(f"Datos cargados:")
print(f"  X: {Xpd.shape} (muestras × genes)")
print(f"  Y: {Ypd.shape} (muestras × vías)")

X = Xpd.values
Y = Ypd.values

cv_exter = KFold(n_splits=5, shuffle=True, random_state=42)
cv_inter = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = np.logspace(-3, 3, 20)

mejores_alphas_global = []
mejores_coeficientes = []
metricas = {}

start_total = time.perf_counter()
# Varianza 0 en vias para todos
variance_selector = VarianceThreshold(threshold=0)
Y = Ypd.values
Y = variance_selector.fit_transform(Y)
indices_vias_filtradas = variance_selector.get_support(indices=True)

vias_filtradas = [vias_inicial[i] for i in indices_vias_filtradas]

print(f"Vías originales: {len(vias_inicial)}")
print(f"Vías después del filtro: {len(vias_filtradas)}")
print(f"Vías eliminadas: {len(vias_inicial) - len(vias_filtradas)}")

for fold, (train_id, test_id) in enumerate(cv_exter.split(X), start=1):
    print(f"\n{'='*50}")
    print(f"FOLD EXTERNO {fold}/5")
    print(f"{'='*50}")
    fold_start = time.perf_counter()

    X_train, X_test = X[train_id], X[test_id]
    Y_train, Y_test = Y[train_id], Y[test_id]
    
    print(f"\nDivisión de datos:")
    print(f"  Train: {X_train.shape[0]} muestras, {X_train.shape[1]} genes")
    print(f"  Test:  {X_test.shape[0]} muestras")

    print(f"\nPreprocesamiento y escalado...")
    
    # Pipeline para X - importante guardar los genes que sobreviven
    preprocessor_X = Pipeline([
        ('variance', VarianceThreshold(threshold=0)),
        ('scaler', StandardScaler())
    ])
    
    X_train_scaled = preprocessor_X.fit_transform(X_train)
    X_test_scaled = preprocessor_X.transform(X_test)
    
    # Obtener los genes que NO fueron eliminados por VarianceThreshold
    # VarianceThreshold elimina features con varianza cero
    var_threshold = preprocessor_X.named_steps['variance']
    genes_filtrados_indices = var_threshold.get_support(indices=True)  # Índices de genes que se mantienen
    genes_filtrados = [genes[i] for i in genes_filtrados_indices]  # Nombres de genes filtrados
    
    print(f"  Genes originales: {X_train.shape[1]}")
    print(f"  Genes después del filtro: {len(genes_filtrados)}")
    print(f"  Genes eliminados: {X_train.shape[1] - len(genes_filtrados)}")
    
    scaler_Y = Pipeline([
         ('scaler', StandardScaler())
    ])
      
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_test_scaled = scaler_Y.transform(Y_test)
    
    print(f"  Vías objetivo: {Y_train_scaled.shape[1]}")

    print(f"\nBuscando alpha óptimo global con RidgeCV...")
    
    ridge_cv = RidgeCV(
        alphas=alphas,
        cv=cv_inter,
        scoring='r2'
    )
    
    ridge_cv.fit(X_train_scaled, Y_train_scaled)
    alpha_global = ridge_cv.alpha_
    
    print(f"  Alpha global seleccionado: {alpha_global:.6f}")
        
    print("INICIO DE PREDICCIÓN")
    Y_pred_scaled = ridge_cv.predict(X_test_scaled)
    
    print("Calculando métricas por vía...")
    fold_metricas = {}    
    count = 0

    for via_idx, via in enumerate(vias_filtradas):
    
        # Valores reales y predichos
        y_test = Y_test_scaled[:, via_idx]
        y_pred = Y_pred_scaled[:, via_idx]

        # Métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Conteo de R² negativos
        if r2 < 0:
            count += 1

        # Guardado de métricas por vía
        fold_metricas[via] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }

    metricas[f'fold_{fold}'] = fold_metricas

    # Resumen
    print(f"Se encontraron {count} vías con R² negativo")
  
    mejores_alphas_global.append(alpha_global)
    
    # Guardar coeficientes con información de genes filtrados
    coef_info = {
        'coeficientes': ridge_cv.coef_,
        'genes_filtrados': genes_filtrados,
        'vias': vias_filtradas
    }
    mejores_coeficientes.append(coef_info)
   
    """Guarda las predicciones de un fold específico"""
    pred_folder = f"{pathway_result}/predicciones"
    os.makedirs(pred_folder, exist_ok=True)
    
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    Y_test_original = scaler_Y.inverse_transform(Y_test_scaled)

    samples_test = [samples[i] for i in test_id]

    df_real = pd.DataFrame(
        Y_test_original,
        index=samples_test,
        columns=vias_filtradas
    )   

    df_pred = pd.DataFrame(
        Y_pred,
        index=samples_test,
        columns=vias_filtradas
    )   

    df_real.to_csv(f"{pred_folder}/fold{fold}_test.csv")
    df_pred.to_csv(f"{pred_folder}/fold{fold}_predicciones.csv")

    fold_end = time.perf_counter()
    fold_time = fold_end - fold_start
    
    print(f"\nRESUMEN FOLD {fold}:")
    print(f"  Alpha global: {alpha_global:.6f}")
    print(f"  Tiempo del fold: {fold_time:.1f} segundos ({fold_time/60:.1f} minutos)")
    
    if fold < 5:
        tiempo_restante = (fold_time * (5 - fold)) / 60
        print(f"  Tiempo estimado restante: {tiempo_restante:.1f} minutos")

end_total = time.perf_counter()
total_time = end_total - start_total
print(f"\n{'='*50}")
print(f"ENTRENAMIENTO COMPLETADO")
print(f"Tiempo total: {total_time/60:.1f} minutos")
print(f"{'='*50}")

# GUARDAR RESULTADOS
print(f"\nGuardando resultados...")

# Guardar métricas externas
with open(f'{pathway_result}/metricas_Rid.json', "w") as f:
    json.dump(metricas, f, indent=4)
print(f"  Métricas guardadas")

# 2. Guardar coeficientes del ÚLTIMO fold (corregido)
if len(mejores_coeficientes) > 0:
    # Tomar el último fold
    coef_info = mejores_coeficientes[-1]
    coeficientes = coef_info['coeficientes']
    genes_filtrados = coef_info['genes_filtrados']
    vias = coef_info['vias']
    
    # Verificar dimensiones
    print(f"\nDimensiones de coeficientes: {coeficientes.shape}")
    print(f"Número de genes filtrados: {len(genes_filtrados)}")
    print(f"Número de vías: {len(vias)}")
    
    # Coeficientes tienen forma (n_vias, n_genes_filtrados)
    # Crear DataFrame correctamente dimensionado
    coef_df = pd.DataFrame(
        coeficientes,  # forma: (n_vias, n_genes_filtrados)
        index=vias_filtradas,    # filas = vías
        columns=genes_filtrados  # columnas = genes filtrados
    )
    
    coef_df.to_csv(f'{pathway_result}/coeficientes_ridge_global.csv')
    print(f"  Coeficientes guardados: {coef_df.shape} (vías × genes)")

# Guardar alphas globales
alphas_df = pd.DataFrame({
    'fold': range(1, len(mejores_alphas_global) + 1),
    'alpha_global': mejores_alphas_global
})
alphas_df.to_csv(f'{pathway_result}/alphas_globales.csv', index=False)
print(f"  Alphas guardados: {len(mejores_alphas_global)} folds")

# Guardar TODOS los coeficientes de todos los folds
coeficientes_completos = []
for i, coef_info in enumerate(mejores_coeficientes, 1):
    fold_data = {
        'fold': i,
        'alpha': mejores_alphas_global[i-1],
        'coeficientes': coef_info['coeficientes'].tolist(),
        'genes': coef_info['genes_filtrados'],
        'vias': coef_info['vias']
    }
    coeficientes_completos.append(fold_data)

with open(f'{pathway_result}/coeficientes_todos_folds.json', 'w') as f:
    json.dump(coeficientes_completos, f, indent=4)
print(f"  Todos los coeficientes guardados en JSON")

# Mostrar resumen final
print(f"\n{'='*50}")
print("RESUMEN FINAL - RIDGE CV GLOBAL")
print(f"{'='*50}")

print(f"\nALPHA GLOBAL POR FOLD:")
for i, alpha in enumerate(mejores_alphas_global, 1):
    print(f"  Fold {i}: {alpha:.6f}")

alpha_mean = np.mean(mejores_alphas_global)
alpha_std = np.std(mejores_alphas_global)
print(f"\nPromedio: {alpha_mean:.6f} ± {alpha_std:.6f}")
print(f"Rango: [{min(mejores_alphas_global):.6f}, {max(mejores_alphas_global):.6f}]")

# Guardar información de tiempo y dimensiones
tiempo_info = {
    'tiempo_total_segundos': total_time,
    'tiempo_total_minutos': total_time / 60,
    'tiempo_promedio_por_fold': total_time / len(mejores_alphas_global),
    'fecha_ejecucion': time.strftime("%Y-%m-%d %H:%M:%S"),
    'dimensiones_originales': {
        'muestras': X.shape[0],
        'genes_originales': X.shape[1],
        'vias': Y.shape[1]
    },
    'dimensiones_filtradas': {
        'genes_filtrados_promedio': len(genes_filtrados) if 'genes_filtrados' in locals() else 'N/A'
    },
    'parametros': {
        'n_folds_externos': cv_exter.n_splits,
        'n_folds_internos': cv_inter.n_splits,
        'n_alphas': len(alphas)
    }
}

print(f"\n{'='*50}")
print(f"Resultados guardados en: {os.path.abspath(pathway_result)}")
print(f"{'='*50}")


