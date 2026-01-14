import os
import sys
import time
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

## RUTA PARA GUARDAR RESULTADOS
pathway_result = "ElasticNet/"
os.makedirs(pathway_result, exist_ok=True)

print("================= MTELASTICNET ==========================")

##CARGA DE DATOS

print("CARGA DE DATOS")
Xpd = pd.read_csv("Data/BLOOD_RNA.csv", index_col=0).sort_index(axis=0)
Ypd = pd.read_csv("Data/hipathia_thyroid.csv", index_col=0).sort_index(axis=0)

print(f"Datos cargados exitosamente: X={Xpd.shape}, Y={Ypd.shape}")

# Guardar nombres antes de convertir
genes = Xpd.columns.tolist()
vias_inicial = Ypd.columns.tolist()
samplesX = Xpd.index.tolist()
samplesY = Ypd.index.tolist()

print(f"Datos cargados:")
print(f"  X: {Xpd.shape} (muestras × genes)")
print(f"  Y: {Ypd.shape} (muestras × vías)")

# Convertir a NumPy arrays
X = Xpd.values
Y = Ypd.values

# CV externa: para evaluar el rendimiento general del modelo
cv_exter = KFold(n_splits=5, shuffle=True, random_state=42)

# CV interna: usada por MultiTaskElasticNetCV
cv_inter = KFold(n_splits=3, shuffle=True, random_state=42)

# Parámetros para MultiTaskElasticNetCV
l1_ratios = [0.1, 0.3, 0.5, 0.9]
alphas = np.logspace(-3, 3, 20)

# Resultados
metricas = {}
todas_predicciones = []
mejores_parametros = []
mejores_rutas = []

start_total = time.perf_counter()

print("INICIANDO CV ANIDADA PARA MULTITASELASTICNET")
print("="*60)

start_total = time.perf_counter() #Inicio para evaluar tiempo

# Varianza 0 en vías para todos, se eliminan las vías
variance_selector = VarianceThreshold(threshold=0)
Y = Ypd.values
Y = variance_selector.fit_transform(Y)
indices_vias_filtradas = variance_selector.get_support(indices=True)

vias_filtradas = [vias_inicial[i] for i in indices_vias_filtradas]

print(f"Vías originales: {len(vias_inicial)}")
print(f"Vías después del filtro: {len(vias_filtradas)}")
print(f"Vías eliminadas: {len(vias_inicial) - len(vias_filtradas)}")

## INICIO CV ANIDADA
for fold, (train_id, test_id) in enumerate(cv_exter.split(X), start=1):
    print(f"\n{'='*50}")
    print(f"FOLD EXTERNO {fold}/5")
    print(f"{'='*50}")
    fold_start = time.perf_counter() # Inicio tiempo por fold

    #Split de X e Y
    X_train, X_test = X[train_id], X[test_id]
    Y_train, Y_test = Y[train_id], Y[test_id]
    
    #Control de datos
    print(f"\nDivisión de datos:")
    print(f"  Train: {X_train.shape[0]} muestras, {X_train.shape[1]} genes")
    print(f"  Test:  {X_test.shape[0]} muestras")

    print(f"\nPreprocesamiento y escalado")
    
    # Pipeline para X
    preprocessor_X = Pipeline([
        ('variance', VarianceThreshold(threshold=0)),
        ('scaler', StandardScaler())
    ])
    
    X_train_scaled = preprocessor_X.fit_transform(X_train)
    X_test_scaled = preprocessor_X.transform(X_test)
    
    # Obtener los genes que NO fueron eliminados por VarianceThreshold
    var_threshold = preprocessor_X.named_steps['variance']
    genes_filtrados_indices = var_threshold.get_support(indices=True)  # Índices de genes que se mantienen
    genes_filtrados = [genes[i] for i in genes_filtrados_indices]  # Nombres de genes filtrados
    
    print(f"  Genes originales: {X_train.shape[1]}")
    print(f"  Genes después del filtro: {len(genes_filtrados)}")
    print(f"  Genes eliminados: {X_train.shape[1] - len(genes_filtrados)}")
    
    # Pipeline para Y
    scaler_Y = Pipeline([
         ('scaler', StandardScaler())
    ])
      
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_test_scaled = scaler_Y.transform(Y_test)
    
    print(f"  Vías objetivo: {Y_train_scaled.shape[1]}")

    # Modelo con CV interna
    modeloEN = MultiTaskElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=cv_inter,
        n_jobs=-1,
        max_iter=3000,
        selection='random',
        random_state=42
    )

    # Entrenar modelo
    print(f"\n  Entrenando con CV interna ({cv_inter.n_splits} folds)...")
    modeloEN.fit(X_train_scaled, Y_train_scaled)
   
    # Mejores Parámetros
    mejor_alpha = modeloEN.alpha_
    mejor_l1 = modeloEN.l1_ratio_
    print(f"  Mejor alpha: {mejor_alpha:.4f}")
    print(f"  Mejor l1_ratio: {mejor_l1:.4f}")

    # Predicción en muestras test
    Y_pred_scaled = modeloEN.predict(X_test_scaled)

    # GUARDAR PREDICCIONES
    pred_folder = f"{pathway_result}/predicciones"
    os.makedirs(pred_folder, exist_ok=True)
    
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    Y_test_original = scaler_Y.inverse_transform(Y_test_scaled)

    # Convertir a DataFrame con nombres de muestras y vías
    samples_test = [samplesY[i] for i in test_id]  
    df_pred = pd.DataFrame(Y_pred, index=samples_test, columns=vias_filtradas)
    df_pred.to_csv(f"{pred_folder}/fold{fold}_predicciones.csv")
    df_test = pd.DataFrame(Y_test_original, index=samples_test, columns=vias_filtradas)
    df_test.to_csv(f"{pred_folder}/fold{fold}_test.csv")

    print("Calculo métricas por vía")
    fold_metricas = {}
    
    count = 0

    for via_idx, via in enumerate(vias_filtradas):
        y_test = Y_test_scaled[:, via_idx]
        y_pred = Y_pred_scaled[:, via_idx]
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)     
        r2 = r2_score(y_test, y_pred)
        
        if r2 < 0:
             count += 1
           
        fold_metricas[via] = {
            'r1': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    metricas[f'fold_{fold}'] = fold_metricas

    print(f"Número de vías con R² negativo: {count}/{len(vias_filtradas)}")

    mejores_parametros.append({'alpha': mejor_alpha, 'l1_ratio': mejor_l1})
    mejores_rutas.append(modeloEN.coef_)

    fold_end = time.perf_counter()
    fold_time = fold_end - fold_start
    
    print(f"\nRESUMEN FOLD {fold}:")

    print(f"  Tiempo del fold: {fold_time:.1f} segundos ({fold_time/60:.1f} minutos)")
  
    print(f"\n{'='*60}")
    print(f"FIN FOLD EXTERNO {fold}")
    print(f"{'='*60}")

end_total = time.perf_counter()
total_time = end_total - start_total
print(f"\n{'='*60}")
print(f"ENTRENAMIENTO COMPLETADO")
print(f"Tiempo total: {total_time/60:.1f} minutos")
print(f"{'='*60}")


print("\nGuardando resultados")

# Guardar métricas
with open(f'{pathway_result}/metricas_externas_new.json', "w") as f:
    json.dump(metricas, f, indent=4)
    
# Guardar coeficientes
mejores_rutas_limpio = [arr.tolist() for arr in mejores_rutas]
with open(f'{pathway_result}/mejores_rutas_new.json', "w") as f:
    json.dump(mejores_rutas_limpio, f, indent=4)

# Guardar parámetros
with open(f'{pathway_result}/mejores_parametros.json', "w") as f:
    json.dump(mejores_parametros, f, indent=4)

print("\nALPHAS Y L1_RATIO SELECCIONADOS POR FOLD:")
for i, parametros in enumerate(mejores_parametros, 1):
    print(f"  Fold {i}: alpha={parametros['alpha']:.6f}, l1_ratio={parametros['l1_ratio']:.3f}")


print("==================== PROCESO COMPLETADO EXITOSAMENTE ==============================")
