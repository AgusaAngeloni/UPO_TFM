import os
import time
import json
import numpy as np
import pandas as pd
import json
import time
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold

print("\n" + "="*50)
print("INICIANDO RANDOMFOREST")
print("="*50)

# RUTA PARA GUARDAR ARCHIVOS
pathway_result = "RandomForest/"
# Crear carpeta si no existe
os.makedirs(pathway_result, exist_ok=True)

print("CARGANDO DATOS")
# Cargar y ordenar las tablas por orden alfabetico para comparar correctamente las predicciones
Xpd = pd.read_csv("Data/BLOOD_RNA.csv", index_col=0).sort_index(axis=0)
Ypd = pd.read_csv("Data/hipathia_thyroid.csv", index_col=0).sort_index(axis=0)

# Guardar nombres antes de convertir
genes = Xpd.columns.tolist()
vias = Ypd.columns.tolist()

print(f"Datos cargados:")
print(f"  X: {Xpd.shape} (muestras × genes)")
print(f"  Y: {Ypd.shape} (muestras × vías)")

samplesY = Ypd.index.tolist()

# Convertir a NumPy arrays
X = Xpd.values  # o Xpd.to_numpy()
Y1 = Ypd.values  # o Ypd.to_numpy()

# Aplicar VarianceThreshold separadamente
var_selector_Y = VarianceThreshold(threshold=0)

Y = var_selector_Y.fit_transform(Y1)

print("Variación de rutas por varianza 0")
print(f"Y: {Y1.shape[1]} → {Y.shape[1]} outputs")

# Obtener nombres de features/vías que sobreviven
vias_filtradas_idx = var_selector_Y.get_support(indices=True)

vias_filtradas = [vias[i] for i in vias_filtradas_idx]

# CV externa: para evaluar el rendimiento general del modelo
cv_exter = KFold(n_splits=5, shuffle=True, random_state=42)

# Resultados
mejores_parametros = []  # best_params_per_fold
importancias_features = []  # feature_importance_per_fold
importancias_features_raw = []
metricas = {}

start_total = time.perf_counter() # Inicio de tiempo de procesamiento

for fold, (train_id, test_id) in enumerate(cv_exter.split(X), start=1): 
    print(f"\n{'='*50}")
    print(f"FOLD EXTERNO {fold}/5")
    print(f"{'='*50}")
    fold_start = time.perf_counter()

    # Split
    X_train, X_test = X[train_id], X[test_id]
    var_selector_X = VarianceThreshold(threshold=0)
    X_train = var_selector_X.fit_transform(X_train)  
    X_test = var_selector_X.transform(X_test)        
    genes_filtrados = [genes[i] for i in var_selector_X.get_support(indices=True)]

    Y_train, Y_test = Y[train_id], Y[test_id]
    
    # Control del split de datos
    print(f"\nDatos del fold:")
    print(f"\tX_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"\tY_train: {Y_train.shape}, Y_test: {Y_test.shape}")
    
    # Modelo RandomForest
    modeloRF = RandomForestRegressor(
        n_estimators=200,
        max_depth= 20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features= 'sqrt',
        n_jobs=-1,
        random_state=42,
        bootstrap=True,
        oob_score=False,
        verbose=0
    )
    
    modeloRF.fit(X_train, Y_train)
        
    print(f"Entrenamiento completado")

    # PREDCICCIONES
    Y_pred = modeloRF.predict(X_test)

    # Guardar predicciones
    pred_folder = f"{pathway_result}/predicciones"
    os.makedirs(pred_folder, exist_ok=True)
    
    # Convertir a DataFrame con nombres de muestras y vías
    samples_test = [samplesY[i] for i in test_id]
    df_pred = pd.DataFrame(Y_pred, index=samples_test, columns=vias_filtradas)
    df_pred.to_csv(f"{pred_folder}/fold{fold}_predicciones.csv")
    df_test = pd.DataFrame(Y_test, index=samples_test, columns=vias_filtradas)
    df_test.to_csv(f"{pred_folder}/fold{fold}_test.csv")

    print("Calculando métricas por vía")
    fold_metricas = {}
    
    count = 0

    for via_idx, via in enumerate(vias_filtradas):
        y_test = Y_test[:, via_idx]
        y_pred = Y_pred[:, via_idx]
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)     
        r2 = r2_score(y_test, y_pred)
        
        if r2 < 0:
             count += 1
           
        fold_metricas[via] = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    metricas[f'fold_{fold}'] = fold_metricas

    print(f"Número de vías con R² negativo: {count}/{len(vias_filtradas)}")
    
    # EXTRAER IMPORTANCIAS CON NOMBRES DE GENES
    print(f"\nExtrayendo importancias de features")

    # Importancia promedio de todos los árboles
    importancias_por_arbol = []
    for tree_idx, tree in enumerate(modeloRF.estimators_):
        importancias_por_arbol.append(tree.feature_importances_)
        if (tree_idx + 1) % 100 == 0:
            print(f"  Procesado árbol {tree_idx + 1}/{len(modeloRF.estimators_)}")

    # Calculo importancia promedio
    importancia_promedio = np.mean(importancias_por_arbol, axis=0)

    # Crear DataFrame con nombres de genes
    df_importancias = pd.DataFrame({
        'gen': genes_filtrados,
        'importancia': importancia_promedio
    })

    # Ordenar por importancia descendente
    df_importancias = df_importancias.sort_values('importancia', ascending=False)

    print(f"\nTop 10 genes más importantes (Fold {fold}):")
    for i, row in df_importancias.head(10).iterrows():
        print(f"  {row['gen']}: {row['importancia']:.6f}")

    # Guardar importancias en ambos formatos
    importancias_features.append(df_importancias)  # DataFrame con nombres
    importancias_features_raw.append(importancia_promedio)  # Array numérico

# Tiempo total
end_total = time.perf_counter()
total_time = end_total - start_total
print(f"\n{'='*60}")
print(f"ENTRENAMIENTO COMPLETADO")
print(f"Tiempo total: {total_time/60:.1f} minutos")
print(f"{'='*60}")

# GUARDAR RESULTADOS
print(f"\nGuardando resultados en: {pathway_result}")

# Guardar métricas
with open(f'{pathway_result}/metricas.json', "w") as f:
    json.dump(metricas, f, indent=4)

for fold_idx, df_imp in enumerate(importancias_features, 1):
    df_imp.to_csv(f'{pathway_result}/importancias_fold_{fold_idx}.csv', index=False)

importancias_combinadas = {}
for fold_idx, df_imp in enumerate(importancias_features, 1):
    importancias_combinadas[f'fold_{fold_idx}'] = df_imp.to_dict('records')

# Guardar importancias
with open(f'{pathway_result}/importancias_todos_folds.json', 'w') as f:
    json.dump(importancias_combinadas, f, indent=4)

# Guardar importancias raw (arrays numéricos)
importancias_raw_limpio = [arr.tolist() for arr in importancias_features_raw]
with open(f'{pathway_result}/importancias_raw.json', "w") as f:
    json.dump(importancias_raw_limpio, f, indent=4)

info_filtrado = {
    'genes_originales': len(genes),
    'genes_filtrados': len(genes_filtrados),
    'vias_originales': len(vias),
    'vias_filtradas': len(vias_filtradas),
    'genes_eliminados': len(genes) - len(genes_filtrados),
    'vias_eliminadas': len(vias) - len(vias_filtradas),
    'genes_filtrados_lista': genes_filtrados,
    'vias_filtradas_lista': vias_filtradas
}

# Guardar Resumen
with open(f'{pathway_result}/info_filtrado.json', 'w') as f:
    json.dump(info_filtrado, f, indent=4)


print(f"TAREA FINALIZADA EXITOSAMENTE")
