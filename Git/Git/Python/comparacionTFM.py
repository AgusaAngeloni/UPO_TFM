import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# FUNCIÓN CARGAR DATOS
def cargar_metricas(path):
    with open(path, 'r') as f:
        data = json.load(f)
    print("Metricas cargadas con éxito")
    return pd.DataFrame(data)

# FUNCIÓN EXTRACCIÓN DE METRICAS
def extraer_modelo_mean(df, metrica):
    mediaModelo = df.applymap(
        lambda x: x.get(metrica) if isinstance(x, dict) else np.nan
    )

    # Forzar tipo numérico
    mediaModelo = mediaModelo.apply(pd.to_numeric, errors="coerce")

    return mediaModelo.mean(axis=1)

# FUNCIÓN PARA FILTRAR RUTAS R2 QUE SE MUESTRAN EN EL GRÁFICO
def filtrar_rutas(r2_a, r2_b, limite_min=-1):
    fuera = (r2_a < limite_min) | (r2_b < limite_min)
    dentro = ~fuera

    return {
        "a_dentro": r2_a[dentro],
        "b_dentro": r2_b[dentro],
        "a_fuera": r2_a[fuera],
        "b_fuera": r2_b[fuera],
        "n_fuera": fuera.sum(),
        "n_total": len(r2_a)
    }

# TABLA PARA TABLA DE VALORES R2 < -1
def tabla_outliers_r2(
    r2_a,
    r2_b,
    nombre_a,
    nombre_b,
    limite_min=-1
):
    fuera = (r2_a < limite_min) | (r2_b < limite_min)

    filas = []

    for ruta in r2_a.index[fuera]:
        val_a = r2_a.loc[ruta]
        val_b = r2_b.loc[ruta]

        if val_a < limite_min and val_b < limite_min:
            razon = f"Ambos < {limite_min}"
        elif val_a < limite_min:
            razon = f"{nombre_a} < {limite_min}"
        else:
            razon = f"{nombre_b} < {limite_min}"

        filas.append({
            "Ruta": ruta,
            f"{nombre_a}_R²": val_a,
            f"{nombre_b}_R²": val_b,
            "Razón": razon
        })

    return (
        pd.DataFrame(filas)
          .sort_values(f"{nombre_a}_R²")
          .reset_index(drop=True)
    )

# FUNCIÓN GRÁFICOS CORRELACIONES
def scatter_r2(
    r2_a,
    r2_b,
    nombre_a,
    nombre_b,
    limite_min=-1,
    limite_max=0.75,
    outpath=None,
    tabla_path=None
):
    datos = filtrar_rutas(r2_a, r2_b, limite_min)
    print("Rutas filtradas con éxito")

    # TABLA DE R < -1
    tabla_outliers = tabla_outliers_r2(
        r2_a,
        r2_b,
        nombre_a,
        nombre_b,
        limite_min
    )

    if tabla_path:
        tabla_outliers.to_csv(tabla_path, index=False)

    # CORRELACIÓN
    corr, pval = pearsonr(
        datos["a_dentro"],
        datos["b_dentro"]
    )

    # GRÁFICO SCATTER
    fig, ax = plt.subplots(figsize=(9,9))

    ax.scatter(
        datos["a_dentro"],
        datos["b_dentro"],
        alpha=0.5,
        s=20
    )

    # Línea roja (identidad)
    ax.plot(
        [limite_min, limite_max],
        [limite_min, limite_max],
        'r--',
        linewidth=2
    )

    # Líneas x=0 e y=0
    ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xlim(limite_min, limite_max)
    ax.set_ylim(limite_min, limite_max)

    ax.set_xlabel(f'R² medio - {nombre_a}', fontsize=12)
    ax.set_ylabel(f'R² medio - {nombre_b}', fontsize=12)

    titulo = (
        f"Comparación de R²: {nombre_a} vs {nombre_b}\n"
        f"n={datos['n_total'] - datos['n_fuera']}/{datos['n_total']}  "
        f"({datos['n_fuera']} fuera del rango)\n"
        f"ρ = {corr:.2f} (p = {pval:.2e})"
    )

    ax.set_title(titulo, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=600, bbox_inches='tight')

    plt.show()

# CARGA DE DATOS
RF_metricas  = cargar_metricas('RandomForest/metricas.json')
E_metricas   = cargar_metricas('ElasticNet/metricas_externas_new.json')
Rid_metricas = cargar_metricas('RidgeCV/metricas_Rid.json')

RF_R2_mean  = extraer_modelo_mean(RF_metricas,'r2')
E_R2_mean   = extraer_modelo_mean(E_metricas,'r2')
Rid_R2_mean = extraer_modelo_mean(Rid_metricas,'r2')

# =========================================================================
#                          GRÁFICOS
# =========================================================================
base_path = "Graph"

# Ridge vs ElasticNet
scatter_r2(
    Rid_R2_mean,
    E_R2_mean,
    "RidgeCV",
    "ElasticNet",
    outpath=f"{base_path}/R2_Ridge_vs_EN.png",
    tabla_path=f"{base_path}/neg_R2_Ridge_vs_EN.csv",
)

# Random Forest vs ElasticNet
scatter_r2(
    RF_R2_mean,
    E_R2_mean,
    "RandomForest",
    "ElasticNet",
    outpath=f"{base_path}/R2_RF_vs_EN.png",
    tabla_path=f"{base_path}/neg_R2_RF_vs_EN.csv",
)

# Ridge vs Random Forest
scatter_r2(
    Rid_R2_mean,
    RF_R2_mean,
    "RidgeCV",
    "RandomForest",
    outpath=f"{base_path}/R2_Ridge_vs_RF.png",
    tabla_path=f"{base_path}/neg_R2_Ridge_vs_RF.csv",
)

df_R2 = pd.DataFrame({
    'RidgeCV':      Rid_R2_mean,
    'ElasticNet':   E_R2_mean,
    'RandomForest': RF_R2_mean
})

# =================== RMSE
RF_RMSE_mean  = extraer_modelo_mean(RF_metricas,'rmse')
E_RMSE_mean   = extraer_modelo_mean(E_metricas,'rmse')
Rid_RMSE_mean = extraer_modelo_mean(Rid_metricas,'rmse')

df_RMSE = pd.DataFrame({
    'RidgeCV':      Rid_RMSE_mean,
    'ElasticNet':   E_RMSE_mean,
    'RandomForest': RF_RMSE_mean
})

# ================== MAE
RF_MAE_mean  = extraer_modelo_mean(RF_metricas,'mae')
E_MAE_mean   = extraer_modelo_mean(E_metricas,'mae')
Rid_MAE_mean = extraer_modelo_mean(Rid_metricas,'mae')

df_MAE = pd.DataFrame({
    'RidgeCV':      Rid_MAE_mean,
    'ElasticNet':   E_MAE_mean,
    'RandomForest': RF_MAE_mean
})

# Formato largo
df_R2_long = (
    df_R2
    .reset_index()                         # Convertir el índice (rutas) en una columna
    .rename(columns={'index': 'Ruta'})     # Renombrar la columna del índice
    .melt(
        id_vars='Ruta',                    # Invertir tabla
        var_name='Modelo',
        value_name='R2'
    )
)

# Identificar las rutas con valores extremos de R2 (R2 <= -1)
rutas_fuera = (
    df_R2_long
    .query("R2 <= -1")                     # Filtrar rutas con R2 muy negativo
    .sort_values("R2")                     # Ordenar
)

# Filtrar las rutas que sí se van a usar para el gráfico
df_R2_plot = df_R2_long[df_R2_long["R2"] > -1]

rutas_fuera.to_csv(f"{base_path}/rutasfuera.csv", index=False)

# ============ GRÁFICO DISTRIBUCIÓN R2
colores = {
    'RidgeCV':      '#2ca02c',   # verde
    'ElasticNet':   '#ff7f0e',  # naranja
    'RandomForest': '#1f77b4'  # azul
}

plt.figure(figsize=(9, 9))

modelos = df_R2_plot['Modelo'].unique()
posiciones = range(1, len(modelos) + 1)

for pos, modelo in zip(posiciones, modelos):
    datos = df_R2_plot[df_R2_plot['Modelo'] == modelo]['R2']

    # Boxplot
    plt.boxplot(
        datos,
        positions=[pos],
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=colores[modelo], alpha=0.4),
        medianprops=dict(color='black')
    )

    # Puntos (jitter manual)
    x_jitter = np.random.normal(pos, 0.04, size=len(datos))
    plt.scatter(
        x_jitter,
        datos,
        color=colores[modelo],
        alpha=0.6,
        s=18
    )

plt.axhline(y=0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
plt.xticks(posiciones, modelos)
plt.ylabel("R²")
plt.title("Comparación de R² por modelo predictivo\n(Rutas con R² > -1)")
plt.grid(axis='y', alpha=0.3)
plt.ylim(-1, 0.75)
plt.savefig(f"{base_path}/comparacionR2.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()

# ============ GRÁFICO DISTRIBUCIÓN MAE

df_MAE_plot = (
    df_MAE
    .reset_index()
    .rename(columns={'index': 'Ruta'})
    .melt(id_vars='Ruta', var_name='Modelo', value_name='MAE')
)

plt.figure(figsize=(9, 9))

modelos = df_MAE_plot['Modelo'].unique()
posiciones = range(1, len(modelos) + 1)

for pos, modelo in zip(posiciones, modelos):
    datos = df_MAE_plot[df_MAE_plot['Modelo'] == modelo]['MAE']

    # Boxplot
    plt.boxplot(
        datos,
        positions=[pos],
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=colores[modelo], alpha=0.4),
        medianprops=dict(color='black')
    )

    # Puntos (jitter manual)
    x_jitter = np.random.normal(pos, 0.04, size=len(datos))
    plt.scatter(
        x_jitter,
        datos,
        color=colores[modelo],
        alpha=0.6,
        s=18
    )

plt.xticks(posiciones, modelos)
plt.ylabel("MAE")
plt.title("DISTRIBUCIÓN DE MAE POR MODELO PREDICTIVO")
plt.grid(axis='y', alpha=0.3)
plt.ylim(-0.05, 1.5)
plt.savefig(f"{base_path}/comparacionRE.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()

# ============ GRÁFICO DISTRIBUCIÓN RMSE

df_RMSE_plot = (
    df_RMSE
    .reset_index()
    .rename(columns={'index': 'Ruta'})
    .melt(id_vars='Ruta', var_name='Modelo', value_name='RMSE')
)

plt.figure(figsize=(9, 9))

modelos = df_RMSE_plot['Modelo'].unique()
posiciones = range(1, len(modelos) + 1)

for pos, modelo in zip(posiciones, modelos):
    datos = df_RMSE_plot[df_RMSE_plot['Modelo'] == modelo]['RMSE']

    # Boxplot
    plt.boxplot(
        datos,
        positions=[pos],
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=colores[modelo], alpha=0.4),
        medianprops=dict(color='black')
    )

    # Puntos (jitter manual)
    x_jitter = np.random.normal(pos, 0.04, size=len(datos))
    plt.scatter(
        x_jitter,
        datos,
        color=colores[modelo],
        alpha=0.6,
        s=18
    )

plt.xticks(posiciones, modelos)
plt.ylabel("RMSE")
plt.title("DISTRIBUCIÓN DE RMSE POR MODELO PREDICTIVO")
plt.grid(axis='y', alpha=0.3)
plt.ylim(-0.05, 1.5)
plt.savefig(f"{base_path}/comparacionRMSE.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()

## ==================== GRÁFICOS HISTOGRAMAS
# ------------ R2 ANTES Y DESPUES DE R2 > 0
def histograma(
    df_todos,
    df_filtrado,
    modelo,
    base_path,
    bins=30,
    color='green'
):
    # --- Datos ---
    r2_antes = df_todos[modelo].dropna()
    r2_despues = df_filtrado.loc[
        df_filtrado["Modelo"] == modelo, "R2"
    ]

    # --- Figura ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Distribución R² – {modelo}', fontsize=16, y=1.02)

    # --- Antes ---
    axes[0].hist(r2_antes, bins=bins, edgecolor='black',
                 alpha=0.7, color=color)
    axes[0].axvline(0, color='red', linestyle='--', label='R² = 0')
    axes[0].set_xlabel('R² promedio')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title(f'Antes del filtro (n={len(r2_antes)})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Después ---
    axes[1].hist(r2_despues, bins=bins, edgecolor='black',
                 alpha=0.7, color=color)
    axes[1].axvline(0, color='red', linestyle='--', label='R² = 0')
    axes[1].set_xlabel('R² promedio')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title(f'Después del filtro (n={len(r2_despues)}) - Eliminadas: {len(r2_antes)-len(r2_despues)}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{base_path}/Hist_R2_{modelo}_antes_despues.png",
        dpi=600,
        bbox_inches='tight'
    )
    plt.show()

modelos = ["RidgeCV", "ElasticNet", "RandomForest"]

for modelo in modelos:
    histograma(
        df_todos= df_R2,     # DF ancho
        df_filtrado= df_R2_plot,  # DF largo
        modelo=modelo,
        base_path=base_path
    )

# --------------------- MAE Y RMSE
def histograma_errores_3modelos(
    df,
    modelos,
    base_path,
    metrica='MAE',
    bins=30,
    color ='pink',
    filename=None
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(
        f'Distribución global de {metrica} por modelo',
        fontsize=16,
        y=1.05
    )

    for ax, modelo in zip(axes, modelos):
        df_mod = df.loc[
            df["Modelo"] == modelo, metrica
        ]

        ax.hist(
            df_mod,
            bins=bins,
            edgecolor='black',
            alpha=0.7,
            color=color
        )

        ax.set_title(f'{modelo}\n(n={len(df_mod)})', fontsize=12)
        ax.set_xlabel(metrica)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('Frecuencia')

    plt.tight_layout()

    if filename:
        plt.savefig(
            f"{base_path}/{filename}",
            dpi=600,
            bbox_inches='tight'
        )

    plt.show()

histograma_errores_3modelos(
    df=df_RMSE_plot,
    modelos=modelos,
    base_path=base_path,
    metrica='RMSE',
    color='blue',
    filename='Hist_RMSE_3modelos.png'
)

histograma_errores_3modelos(
    df=df_MAE_plot,
    modelos=modelos,
    base_path=base_path,
    metrica='MAE',
    filename='Hist_MAE_3modelos.png'
)

## ============ COMPARACIÓN DE WILCOXON
def volcano_plot_wilcoxon(
    df,
    fdr_threshold=0.05,
    stat_threshold=0,
    title="Volcano plot",
    subtitle=None,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    df = df.copy()
    df["Neg_Log10_FDR"] = -np.log10(df["FDRp.value"])

    # Clasificación
    df["Clase"] = "No Significativa"

    df.loc[
        (df["FDRp.value"] < fdr_threshold) &
        (df["statistic"] > stat_threshold),
        "Clase"
    ] = "Activada en Tiroides (Thyroid)"

    df.loc[
        (df["FDRp.value"] < fdr_threshold) &
        (df["statistic"] < -stat_threshold),
        "Clase"
    ] = "Activada en Sangre (Blood)"

    colors = {
        "Activada en Tiroides (Thyroid)": "red",
        "Activada en Sangre (Blood)": "blue",
        "No Significativa": "gray"
    }

    # Scatter
    for clase, color in colors.items():
        subset = df[df["Clase"] == clase]
        ax.scatter(
            subset["statistic"],
            subset["Neg_Log10_FDR"],
            s=30,
            alpha=0.8,
            label=clase,
            color=color
        )

    # Umbrales
    ax.axhline(-np.log10(fdr_threshold), linestyle="dashed", color="black")
    ax.axvline(stat_threshold, linestyle="dotted", color="green")
    ax.axvline(-stat_threshold, linestyle="dotted", color="green")

    ax.set_xlabel("Cambio relativo de actividad de rutas (Thyroid vs Blood)")
    ax.set_ylabel(r"$-log_{10}$(FDR corregido)")

    ax.set_title(title, fontsize=12, fontweight="bold")
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center")

    # CONTEOS (esto es lo nuevo)
    n_thyroid = (df["Clase"] == "Activada en Tiroides (Thyroid)").sum()
    n_blood = (df["Clase"] == "Activada en Sangre (Blood)").sum()
    n_nosig = (df["Clase"] == "No Significativa").sum()

    texto = (
        f"Thyroid: {n_thyroid}\n"
        f"Blood: {n_blood}\n"
        f"No sig.: {n_nosig}"
    )

    ax.text(
        0.02, 0.98,
        texto,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.legend(frameon=False)

    return ax

def volcano_true_vs_pred_por_modelo(
    base_path,
    fdr_threshold=0.05,
    out_dir=None
):
    if out_dir is None:
        out_dir = base_path

    modelos = sorted(os.listdir(base_path))

    for modelo in modelos:
        path_modelo = os.path.join(base_path, modelo)
        if not os.path.isdir(path_modelo):
            continue

        # detectar folds
        files = os.listdir(path_modelo)
        folds = sorted(
            set(re.findall(r"(fold\d+)_wilcoxon_true\.csv", " ".join(files))),
            key=lambda x: int(x.replace("fold", ""))
        )

        for fold in folds:
            print(f"Procesando {modelo} – {fold}")

            path_true = os.path.join(path_modelo, f"{fold}_wilcoxon_true.csv")
            path_pred = os.path.join(path_modelo, f"{fold}_wilcoxon_pred.csv")

            df_true = pd.read_csv(path_true, index_col=0)
            df_pred = pd.read_csv(path_pred, index_col=0)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            volcano_plot_wilcoxon(
                df_true,
                fdr_threshold=fdr_threshold,
                title="REAL\n",
                subtitle="Wilcoxon Thyroid vs Blood",
                ax=axes[0]
            )

            volcano_plot_wilcoxon(
                df_pred,
                fdr_threshold=fdr_threshold,
                title="PREDICCION\n",
                subtitle="Wilcoxon Thyroid vs Blood",
                ax=axes[1]
            )

            plt.suptitle(
                f"{modelo} – {fold}",
                fontsize=14,
                fontweight="bold"
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            out_file = os.path.join(
                out_dir,
                f"volcano_{modelo}_{fold}_true_vs_pred.png"
            )

            plt.savefig(out_file, dpi=300)
            plt.close()

wilcoxon_path = "Data/wilcoxon"

volcano_true_vs_pred_por_modelo(
    base_path=wilcoxon_path,
    fdr_threshold=0.05,
    out_dir=base_path
)


from matplotlib_venn import venn2
from matplotlib.patches import Patch
from collections import Counter

# =========================
# COLORES
# =========================
COLOR_UP_REAL    = "orange"
COLOR_UP_PRED    = "red"
COLOR_DOWN_REAL  = "green"
COLOR_DOWN_PRED  = "blue"
COLOR_INTER_UP      = "pink"
COLOR_INTER_DOWN      = "lightblue"

# =========================
# CLASIFICAR RUTAS
# =========================
def clasificar_rutas(df, fdr=0.05):
    up = set(df[(df["FDRp.value"] < fdr) & (df["statistic"] > 0)].index)
    down = set(df[(df["FDRp.value"] < fdr) & (df["statistic"] < 0)].index)
    ns = set(df.index) - up - down
    print("Rutas Clasificadas")
    return up, down, ns


# =========================
# CONSENSO ENTRE FOLDS
# =========================
def consenso(lista_sets, min_folds):
    c = Counter()
    for s in lista_sets:
        c.update(s)
    return {k for k, v in c.items() if v >= min_folds}


# =========================
# VENN INDIVIDUAL
# =========================
def venn_signo(ax, real_set, pred_set, titulo,
               color_real, color_pred, COLOR_INTER):

    if len(real_set) == 0 and len(pred_set) == 0:
        ax.text(0.5, 0.5, "Sin rutas significativas",
                ha="center", va="center", fontsize=11)
        ax.set_title(titulo, fontweight="bold")
        ax.set_axis_off()
        return

    solo_real = real_set - pred_set
    solo_pred = pred_set - real_set
    inter     = real_set & pred_set

    v = venn2(
        subsets=(len(solo_real), len(solo_pred), len(inter)),
        set_labels=("Real", "Predicción"),
        ax=ax
    )

    # colores
    if v.get_patch_by_id("10"):
        v.get_patch_by_id("10").set_color(color_real)
        v.get_patch_by_id("10").set_alpha(0.5)

    if v.get_patch_by_id("01"):
        v.get_patch_by_id("01").set_color(color_pred)
        v.get_patch_by_id("01").set_alpha(0.5)

    if v.get_patch_by_id("11"):
        v.get_patch_by_id("11").set_color(COLOR_INTER)
        v.get_patch_by_id("11").set_alpha(0.65)

    # mover labels hacia afuera
    if v.set_labels[0]:
        v.set_labels[0].set_position((-0.5, -0.3))
        v.set_labels[0].set_fontsize(11)

    if v.set_labels[1]:
        v.set_labels[1].set_position((0.5, -0.3))
        v.set_labels[1].set_fontsize(11)

    ax.set_title(titulo, fontweight="bold")

# =========================
# FUNCIÓN PRINCIPAL
# =========================
def venn_por_modelo(base_path,
                    out_path,            
                    fdr=0.05,
                    min_folds=3,
                    guardar=True):

    for modelo in sorted(os.listdir(base_path)):
        modelo_path = os.path.join(base_path, modelo)
        if not os.path.isdir(modelo_path):
            continue

        real_up_folds, real_down_folds = [], []
        pred_up_folds, pred_down_folds = [], []

        for fname in os.listdir(modelo_path):
            if not fname.endswith("_true.csv"):
                continue

            fold = fname.split("_")[0]
            path_true = os.path.join(modelo_path, f"{fold}_wilcoxon_true.csv")
            path_pred = os.path.join(modelo_path, f"{fold}_wilcoxon_pred.csv")

            if not os.path.exists(path_pred):
                continue

            df_true = pd.read_csv(path_true, index_col=0)
            df_pred = pd.read_csv(path_pred, index_col=0)

            ru, rd, _ = clasificar_rutas(df_true, fdr)
            pu, pdn, _ = clasificar_rutas(df_pred, fdr)

            real_up_folds.append(ru)
            real_down_folds.append(rd)
            pred_up_folds.append(pu)
            pred_down_folds.append(pdn)

        # consenso
        real_up   = consenso(real_up_folds, min_folds)
        real_down = consenso(real_down_folds, min_folds)
        pred_up   = consenso(pred_up_folds, min_folds)
        pred_down = consenso(pred_down_folds, min_folds)
        print("Datos listos para graficarse")
        # figura
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        venn_signo(
            axes[0],
            real_up,
            pred_up,
            "Rutas UP",
            COLOR_UP_REAL,
            COLOR_UP_PRED,
            COLOR_INTER_UP
        )

        venn_signo(
            axes[1],
            real_down,
            pred_down,
            "Rutas DOWN",
            COLOR_DOWN_REAL,
            COLOR_DOWN_PRED,
            COLOR_INTER_DOWN
        )

        # leyenda
        legend = [
            Patch(facecolor=COLOR_UP_REAL, alpha=0.5, label="UP real"),
            Patch(facecolor=COLOR_UP_PRED, alpha=0.5, label="UP predicción"),
            Patch(facecolor=COLOR_DOWN_REAL, alpha=0.5, label="DOWN real"),
            Patch(facecolor=COLOR_DOWN_PRED, alpha=0.5, label="DOWN predicción"),
            Patch(facecolor=COLOR_INTER_UP, alpha=0.5,label="Intersección UP"),
            Patch(facecolor=COLOR_INTER_DOWN, alpha=0.5,label="Intersección DOWN")
                  
        ]

        fig.legend(handles=legend, loc="lower center", ncol=3)
        fig.suptitle(f"Preservación del signo de activación de rutas en las predicciones\n{modelo}",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0.1, 1, 0.92])

        if guardar:
            out = os.path.join(out_path,
                               f"venn_{modelo}_consenso.png")
            plt.savefig(out, dpi=300)

        plt.show()

venn_por_modelo(
    base_path="Data/wilcoxon",
    out_path="Graph",
    fdr=0.05,
    min_folds=5,
    guardar=True
)
