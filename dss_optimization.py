import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="DSS Optimización de Producción", layout="wide")

# ============================================================================
# CARGAR DATOS
# ============================================================================
@st.cache_data
def load_data():
    return pd.read_csv('manufacturing_defect_dataset.csv')

df = load_data()

# ============================================================================
# SIDEBAR - NAVEGACIÓN (MENÚ)
# ============================================================================
st.sidebar.title("🏭 DSS Optimización de Producción")
st.sidebar.markdown("---")
section = st.sidebar.selectbox(
    "Menú Principal",
    ["🏠 Inicio",
     "📊 A. Análisis Exploratorio",
     "🔧 B. Preprocessing de Datos",
     "📈 C. Modelado Multiobjetivo",
     "🔮 D. Predicción de Cambios"]
)
st.sidebar.markdown("---")
st.sidebar.info(f"Dataset: {df.shape[0]} registros × {df.shape[1]} variables")

# ============================================================================
# SECCIÓN: INICIO
# ============================================================================
if section == "🏠 Inicio":
    st.title("🏠 Decision Support System - Optimización de Producción")

    st.markdown("""
    ### Objetivo
    Optimizar **costos** y **eficiencia** simultáneamente sin sacrificar **calidad**.

    ### Estructura del análisis
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("A. Análisis Exploratorio")
        st.write("""
        - Estadísticas descriptivas
        - Matriz de correlaciones
        - Visualización de trade-offs
        """)

    with col2:
        st.subheader("B. Preprocessing")
        st.write("""
        - Normalización de variables
        - Métricas compuestas
        - Detección de outliers
        """)

    with col3:
        st.subheader("C. Modelado Multiobjetivo")
        st.write("""
        - Frontera de Pareto
        - Scoring combinado
        - Recomendaciones
        """)

    st.markdown("---")
    st.subheader("Descripción del dataset")

    # Agrupar variables por categoría
    categories = {
        "Costos": ["ProductionCost", "EnergyConsumption", "AdditiveMaterialCost"],
        "Eficiencia": ["WorkerProductivity", "EnergyEfficiency", "InventoryTurnover", "ProductionVolume"],
        "Riesgo": ["DowntimePercentage", "DefectRate"],
        "Control": ["MaintenanceHours", "SupplierQuality", "DeliveryDelay"],
        "Otros": ["QualityScore", "SafetyIncidents", "StockoutRate", "AdditiveProcessTime", "DefectStatus"]
    }

    for category, variables in categories.items():
        with st.expander(category):
            for var in variables:
                if var in df.columns:
                    st.write(f"**{var}**: {df[var].describe()['mean']:.2f} (media)")


# ============================================================================
# SECCIÓN A: ANÁLISIS EXPLORATORIO
# ============================================================================
elif section == "📊 A. Análisis Exploratorio":
    st.title("📊 A. Análisis Exploratorio")

    # Subsecciones
    subsection = st.tabs(["Estadísticas", "Correlaciones", "Distribuciones", "Trade-offs"])

    # TAB: Estadísticas descriptivas
    with subsection[0]:
        st.subheader("Estadísticas Descriptivas")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Tabla resumen de variables críticas
            critical_vars = [
                "ProductionCost", "EnergyConsumption", "AdditiveMaterialCost",
                "WorkerProductivity", "EnergyEfficiency", "InventoryTurnover",
                "DefectRate", "DowntimePercentage", "MaintenanceHours"
            ]

            stats = df[critical_vars].describe().T
            st.dataframe(stats, use_container_width=True)

        with col2:
            st.metric("Total registros", df.shape[0])
            st.metric("Variables", df.shape[1])
            st.metric("Defectos (promedio)", f"{df['DefectRate'].mean():.2f}%")
            st.metric("Costo total (promedio)", f"${df['ProductionCost'].mean():.0f}")

    # TAB: Matriz de correlaciones
    with subsection[1]:
        st.subheader("Matriz de Correlaciones (Variables Clave)")

        # Seleccionar variables relevantes
        corr_vars = [
            "ProductionCost", "EnergyConsumption", "AdditiveMaterialCost",
            "WorkerProductivity", "EnergyEfficiency", "InventoryTurnover",
            "DefectRate", "DowntimePercentage", "MaintenanceHours",
            "SupplierQuality", "ProductionVolume"
        ]

        corr_matrix = df[corr_vars].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    square=True, ax=ax, cbar_kws={'label': 'Correlación'})
        ax.set_title("Matriz de Correlaciones", fontsize=14, fontweight='bold')
        st.pyplot(fig)

        # Interpretación automática
        st.subheader("Insights clave")

        # Correlaciones con DefectRate
        defect_corr = corr_matrix['DefectRate'].sort_values(ascending=False)
        st.write("**Factores que MÁS influyen en tasa de defectos:**")
        st.write(defect_corr.head(5))

        # Correlaciones con costo
        cost_corr = corr_matrix['ProductionCost'].sort_values(ascending=False)
        st.write("**Factores que MÁS influyen en costo de producción:**")
        st.write(cost_corr.head(5))

    # TAB: Distribuciones
    with subsection[2]:
        st.subheader("Distribuciones de Variables Críticas")

        var_to_plot = st.multiselect(
            "Selecciona variables para visualizar:",
            ["ProductionCost", "DefectRate", "WorkerProductivity",
             "EnergyConsumption", "DowntimePercentage", "EnergyEfficiency"],
            default=["ProductionCost", "DefectRate", "WorkerProductivity"]
        )

        if var_to_plot:
            fig, axes = plt.subplots(1, len(var_to_plot), figsize=(15, 4))
            if len(var_to_plot) == 1:
                axes = [axes]

            for ax, var in zip(axes, var_to_plot):
                ax.hist(df[var], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_title(var)
                ax.set_xlabel("Valor")
                ax.set_ylabel("Frecuencia")
                ax.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

    # TAB: Trade-offs
    with subsection[3]:
        st.subheader("Análisis de Trade-offs")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Trade-off 1: Costo vs Eficiencia**")
            fig, ax = plt.subplots(figsize=(8, 5))

            # Crear score simple de eficiencia (normalizado)
            efficiency_score = (
                df['WorkerProductivity'] +
                df['EnergyEfficiency'] * 10 +
                df['InventoryTurnover']
            ) / 3

            scatter = ax.scatter(df['ProductionCost'], efficiency_score,
                               c=df['DefectRate'], cmap='RdYlGn_r', alpha=0.6, s=50)
            ax.set_xlabel("Costo de Producción")
            ax.set_ylabel("Score de Eficiencia")
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Tasa de Defectos")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.write("**Trade-off 2: Mantenimiento vs Downtime**")
            fig, ax = plt.subplots(figsize=(8, 5))

            scatter = ax.scatter(df['MaintenanceHours'], df['DowntimePercentage'],
                               c=df['DefectRate'], cmap='RdYlGn_r', alpha=0.6, s=50)
            ax.set_xlabel("Horas de Mantenimiento")
            ax.set_ylabel("Porcentaje de Downtime")
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Tasa de Defectos")
            ax.grid(alpha=0.3)
            st.pyplot(fig)


# ============================================================================
# SECCIÓN B: PREPROCESSING DE DATOS
# ============================================================================
elif section == "🔧 B. Preprocessing de Datos":
    st.title("🔧 B. Preprocessing de Datos")

    subsection = st.tabs(["Normalización", "Métricas Compuestas"])

    # TAB: Normalización
    with subsection[0]:
        st.subheader("Normalización de Variables (0-100)")

        scaler = MinMaxScaler(feature_range=(0, 100))

        # Variables a normalizar
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]),
            columns=numeric_cols
        )

        st.write("**Primeras filas normalizadas:**")
        st.dataframe(df_normalized.head(10), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Antes de normalizar (ProductionCost):**")
            st.write(f"Mín: ${df['ProductionCost'].min():.2f}")
            st.write(f"Máx: ${df['ProductionCost'].max():.2f}")
            st.write(f"Media: ${df['ProductionCost'].mean():.2f}")

        with col2:
            st.write("**Después de normalizar (ProductionCost):**")
            st.write(f"Mín: {df_normalized['ProductionCost'].min():.2f}")
            st.write(f"Máx: {df_normalized['ProductionCost'].max():.2f}")
            st.write(f"Media: {df_normalized['ProductionCost'].mean():.2f}")

    # TAB: Métricas compuestas
    with subsection[1]:
        st.subheader("Cálculo de Métricas Compuestas")

        # Crear métricas
        df_processed = df.copy()
        df_processed_norm = df_normalized.copy()

        # 1. Costo Total (suma de costos normalizados)
        df_processed['CostoTotal'] = (
            df_normalized['ProductionCost'] * 0.4 +
            df_normalized['EnergyConsumption'] * 0.35 +
            df_normalized['AdditiveMaterialCost'] * 0.25
        )

        # 2. Score de Eficiencia
        df_processed['ScoreEficiencia'] = (
            df_normalized['WorkerProductivity'] * 0.4 +
            df_normalized['EnergyEfficiency'] * 0.3 +
            df_normalized['InventoryTurnover'] * 0.3
        )

        # 3. Índice de Calidad (inverso de defectos, normalizado)
        df_processed['IndiceCalidad'] = (
            (100 - df_normalized['DefectRate']) * 0.5 +
            df_normalized['QualityScore'] * 0.5
        )

        # 4. Costo de Oportunidad (defectos + downtime)
        df_processed['CostoOportunidad'] = (
            df_normalized['DefectRate'] * 0.6 +
            df_normalized['DowntimePercentage'] * 0.4
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Costo Total (ponderado)**")
            st.bar_chart(df_processed['CostoTotal'].head(20))

            st.metric("Costo Promedio", f"{df_processed['CostoTotal'].mean():.2f}")

        with col2:
            st.write("**Score de Eficiencia**")
            st.bar_chart(df_processed['ScoreEficiencia'].head(20))

            st.metric("Eficiencia Promedio", f"{df_processed['ScoreEficiencia'].mean():.2f}")

        st.write("---")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Índice de Calidad**")
            st.bar_chart(df_processed['IndiceCalidad'].head(20))

            st.metric("Calidad Promedio", f"{df_processed['IndiceCalidad'].mean():.2f}")

        with col2:
            st.write("**Costo de Oportunidad**")
            st.bar_chart(df_processed['CostoOportunidad'].head(20))

            st.metric("Oportunidad Promedio", f"{df_processed['CostoOportunidad'].mean():.2f}")

        # Tabla resumen
        st.subheader("Resumen de Métricas Compuestas")
        summary_metrics = df_processed[['CostoTotal', 'ScoreEficiencia', 'IndiceCalidad', 'CostoOportunidad']].describe()
        st.dataframe(summary_metrics, use_container_width=True)


# ============================================================================
# SECCIÓN C: MODELADO MULTIOBJETIVO
# ============================================================================
elif section == "📈 C. Modelado Multiobjetivo":
    st.title("📈 C. Modelado Multiobjetivo - Ranking de Escenarios Optimizados")

    # Cálculo de scoring
    scaler = MinMaxScaler(feature_range=(0, 100))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_norm = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols
    )

    # Scoring multiobjetivo
    df_scoring = pd.DataFrame({
        'Costo': 100 - df_norm['ProductionCost'] * 0.4 - df_norm['EnergyConsumption'] * 0.35 - df_norm['AdditiveMaterialCost'] * 0.25,
        'Eficiencia': df_norm['WorkerProductivity'] * 0.4 + df_norm['EnergyEfficiency'] * 0.3 + df_norm['InventoryTurnover'] * 0.3,
        'Calidad': (100 - df_norm['DefectRate']) * 0.5 + df_norm['QualityScore'] * 0.5
    })

    df_scoring['Score_Combinado'] = (
        df_scoring['Costo'] * 0.35 +
        df_scoring['Eficiencia'] * 0.35 +
        df_scoring['Calidad'] * 0.30
    )

    # Agregar índice del escenario original
    df_scoring['Escenario_ID'] = range(len(df_scoring))

    # Calcular promedios para comparación
    avg_cost = df['ProductionCost'].mean()
    avg_energy = df['EnergyConsumption'].mean()
    avg_material = df['AdditiveMaterialCost'].mean()
    avg_defect = df['DefectRate'].mean()
    avg_productivity = df['WorkerProductivity'].mean()
    avg_efficiency = df['EnergyEfficiency'].mean()
    avg_inventory = df['InventoryTurnover'].mean()
    avg_maintenance = df['MaintenanceHours'].mean()
    avg_downtime = df['DowntimePercentage'].mean()
    avg_quality = df['QualityScore'].mean()
    avg_volume = df['ProductionVolume'].mean()

    # TOP 10 ESCENARIOS CON RESTRICCIONES DE NEGOCIO
    st.subheader("Top 10 Mejores Escenarios (Balance Óptimo)")

    # RESTRICCIONES DE REALIDAD OPERATIVA
    st.markdown("### Restricciones Aplicadas")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_maintenance = st.slider(
            "Mantenimiento mínimo (horas)",
            min_value=0, max_value=23, value=5,
            help="No recomendamos escenarios con menos de este mantenimiento"
        )
    with col2:
        max_defect = st.slider(
            "Defectos máximos aceptables (%)",
            min_value=0.0, max_value=5.0, value=3.5,
            help="Excluir escenarios con tasa de defectos mayor a esto"
        )
    with col3:
        min_quality = st.slider(
            "Quality Score mínimo",
            min_value=0, max_value=100, value=70,
            help="No recomendamos escenarios por debajo de este score de calidad"
        )

    # Aplicar restricciones
    df_filtered = df_scoring.copy()
    df_filtered['Escenario_ID'] = range(len(df))

    # Agregar datos necesarios para filtrado
    df_filtered['MaintenanceHours'] = df['MaintenanceHours'].values
    df_filtered['DefectRate'] = df['DefectRate'].values
    df_filtered['QualityScore'] = df['QualityScore'].values

    # Filtrar según restricciones
    df_filtered = df_filtered[
        (df_filtered['MaintenanceHours'] >= min_maintenance) &
        (df_filtered['DefectRate'] <= max_defect) &
        (df_filtered['QualityScore'] >= min_quality)
    ]

    if len(df_filtered) == 0:
        st.error("No hay escenarios que cumplan con estas restricciones. Ajusta los parámetros.")
        top_10_idx = df_scoring.nlargest(10, 'Score_Combinado')['Escenario_ID'].values
        st.info("Mostrando Top 10 sin restricciones...")
    else:
        top_10_idx = df_filtered.nlargest(10, 'Score_Combinado')['Escenario_ID'].values
        st.success(f"{len(df_filtered)} escenarios cumplen con las restricciones")

    for rank, scenario_id in enumerate(top_10_idx, 1):
        # Datos del escenario
        scenario = df.iloc[scenario_id]
        scenario_scores = df_scoring.iloc[scenario_id]

        with st.expander(f"**#{rank}** - Escenario {scenario_id + 1} | Score Global: {scenario_scores['Score_Combinado']:.1f}/100"):

            # Grid de información
            col1, col2, col3 = st.columns(3)

            # COLUMNA 1: COSTOS
            with col1:
                st.markdown("### Costos")

                cost_total = scenario['ProductionCost'] + scenario['EnergyConsumption'] + scenario['AdditiveMaterialCost']
                avg_total = avg_cost + avg_energy + avg_material

                cost_diff_pct = ((cost_total - avg_total) / avg_total) * 100
                cost_savings = avg_total - cost_total

                st.metric(
                    "Costo Total",
                    f"${cost_total:,.0f}",
                    f"{cost_diff_pct:.1f}% vs promedio",
                    delta_color="inverse"
                )

                st.write(f"**Desglose:**")
                st.write(f"  • Producción: ${scenario['ProductionCost']:,.0f} ({((scenario['ProductionCost']-avg_cost)/avg_cost)*100:+.1f}%)")
                st.write(f"  • Energía: ${scenario['EnergyConsumption']:,.0f} ({((scenario['EnergyConsumption']-avg_energy)/avg_energy)*100:+.1f}%)")
                st.write(f"  • Materiales: ${scenario['AdditiveMaterialCost']:,.0f} ({((scenario['AdditiveMaterialCost']-avg_material)/avg_material)*100:+.1f}%)")

                if cost_savings > 0:
                    st.success(f"**Ahorro estimado: ${cost_savings:,.0f}** por lote")
                else:
                    st.warning(f"Costo adicional: ${abs(cost_savings):,.0f} por lote")

            # COLUMNA 2: CALIDAD
            with col2:
                st.markdown("### Calidad")

                quality_avg = (scenario['QualityScore'] + (100 - scenario['DefectRate'])) / 2

                st.metric(
                    "Índice de Calidad",
                    f"{quality_avg:.0f}/100",
                    f"Defectos: {scenario['DefectRate']:.2f}%"
                )

                defect_diff_pct = ((scenario['DefectRate'] - avg_defect) / avg_defect) * 100

                st.write(f"**Detalles:**")
                st.write(f"  • Tasa de Defectos: {scenario['DefectRate']:.2f}% ({defect_diff_pct:+.1f}% vs promedio)")
                st.write(f"  • Quality Score: {scenario['QualityScore']:.1f}/100 ({((scenario['QualityScore']-avg_quality)/avg_quality)*100:+.1f}%)")
                st.write(f"  • Downtime: {scenario['DowntimePercentage']:.2f}%")

                if scenario['DefectRate'] < avg_defect * 0.8:
                    st.success("Excelente control de defectos")
                elif scenario['DefectRate'] > avg_defect * 1.2:
                    st.error("Defectos elevados")
                else:
                    st.info("Defectos normales")

            # COLUMNA 3: EFICIENCIA
            with col3:
                st.markdown("### Eficiencia")

                efficiency_score = (scenario['WorkerProductivity'] + scenario['EnergyEfficiency']*10 + scenario['InventoryTurnover']) / 3

                st.metric(
                    "Score de Eficiencia",
                    f"{scenario_scores['Eficiencia']:.0f}/100",
                    f"Productividad: {scenario['WorkerProductivity']:.1f}%"
                )

                st.write(f"**Componentes:**")
                st.write(f"  • Productividad: {scenario['WorkerProductivity']:.1f}% ({((scenario['WorkerProductivity']-avg_productivity)/avg_productivity)*100:+.1f}%)")
                st.write(f"  • Eficiencia Energética: {scenario['EnergyEfficiency']:.2f} ({((scenario['EnergyEfficiency']-avg_efficiency)/avg_efficiency)*100:+.1f}%)")
                st.write(f"  • Rotación Inventario: {scenario['InventoryTurnover']:.1f} ({((scenario['InventoryTurnover']-avg_inventory)/avg_inventory)*100:+.1f}%)")
                st.write(f"  • Volumen: {scenario['ProductionVolume']:.0f} unidades ({((scenario['ProductionVolume']-avg_volume)/avg_volume)*100:+.1f}%)")

            # VARIABLES DE CONTROL
            st.markdown("---")
            st.markdown("### Variables de Control")

            control_col1, control_col2, control_col3 = st.columns(3)

            with control_col1:
                maint_diff = ((scenario['MaintenanceHours'] - avg_maintenance) / avg_maintenance) * 100
                st.write(f"**Mantenimiento:** {scenario['MaintenanceHours']:.0f}h ({maint_diff:+.1f}%)")
                st.write(f"  Promedio: {avg_maintenance:.0f}h")

            with control_col2:
                supplier_diff = scenario['SupplierQuality'] - 90
                st.write(f"**Calidad Proveedor:** {scenario['SupplierQuality']:.1f}/100")
                if scenario['SupplierQuality'] > 95:
                    st.write("  Excelente")
                elif scenario['SupplierQuality'] > 85:
                    st.write("  Bueno")
                else:
                    st.write("  Revisar")

            with control_col3:
                delivery_diff = scenario['DeliveryDelay'] - df['DeliveryDelay'].mean()
                st.write(f"**Retraso Entrega:** {scenario['DeliveryDelay']:.0f} días")
                if scenario['DeliveryDelay'] <= 2:
                    st.write("  Excelente")
                elif scenario['DeliveryDelay'] <= 4:
                    st.write("  Normal")
                else:
                    st.write("  Crítico")

            # SCORING DESGLOSADO
            st.markdown("---")
            st.markdown("### Scoring Detallado")

            score_col1, score_col2, score_col3, score_col4 = st.columns(4)

            with score_col1:
                st.metric("Score Costo", f"{scenario_scores['Costo']:.0f}/100")

            with score_col2:
                st.metric("Score Eficiencia", f"{scenario_scores['Eficiencia']:.0f}/100")

            with score_col3:
                st.metric("Score Calidad", f"{scenario_scores['Calidad']:.0f}/100")

            with score_col4:
                st.metric("Score Global", f"{scenario_scores['Score_Combinado']:.0f}/100", "Ponderado")

            # RECOMENDACIÓN
            st.markdown("---")
            st.markdown("### Recomendación Estratégica")

            recommendation = ""
            priority = ""

            # Lógica de recomendación
            if cost_savings > 2000 and scenario['DefectRate'] < avg_defect:
                recommendation = "**IMPLEMENTACIÓN RECOMENDADA** - Este escenario ofrece ahorros significativos SIN comprometer calidad"
                priority = "ALTA PRIORIDAD"
            elif scenario_scores['Calidad'] > 85 and scenario['DefectRate'] < 2:
                recommendation = "**EXCELENTE CALIDAD** - Ideal si tu prioridad es minimizar defectos y retrabajo"
                priority = "PRIORIDAD MEDIA"
            elif scenario_scores['Eficiencia'] > 80:
                recommendation = "**ALTA EFICIENCIA** - Maximiza productividad, requiere inversión en operaciones"
                priority = "PRIORIDAD MEDIA"
            else:
                recommendation = "**BALANCE ACEPTABLE** - Buen equilibrio general, monitorear en ejecución"
                priority = "PRIORIDAD BAJA"

            st.write(f"**{priority}**")
            st.write(recommendation)

            # Acciones requeridas
            if scenario['MaintenanceHours'] > avg_maintenance * 1.3:
                st.warning(f"Requiere aumentar mantenimiento en {scenario['MaintenanceHours'] - avg_maintenance:.0f}h/período")

            if scenario['SupplierQuality'] > 95:
                st.info("Depende de proveedores de alta calidad - negociar contratos SLA")

            if cost_savings > 0:
                st.success(f"Ahorro proyectado: ${cost_savings * 12:,.0f}/año (si se mantiene constante)")

    # VISUALIZACIÓN COMPARATIVA
    st.markdown("---")
    st.subheader("Comparativa Visual: Top 10 vs Promedio")

    top_10_data = df.iloc[top_10_idx]

    comparison_fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    comparison_fig.suptitle("Comparativa: Top 10 Escenarios vs Promedio del Dataset", fontsize=14, fontweight='bold')

    # Plot 1: Costo
    axes[0, 0].barh(['Promedio'], [avg_cost], color='lightcoral', label='Promedio')
    axes[0, 0].barh(['Top 10 (Mejor)'], [top_10_data['ProductionCost'].min()], color='lightgreen')
    axes[0, 0].barh(['Top 10 (Peor)'], [top_10_data['ProductionCost'].max()], color='lightyellow')
    axes[0, 0].set_xlabel("Costo Producción ($)")
    axes[0, 0].set_title("Costo de Producción")
    axes[0, 0].legend()

    # Plot 2: Defectos
    axes[0, 1].barh(['Promedio'], [avg_defect], color='lightcoral')
    axes[0, 1].barh(['Top 10 (Mejor)'], [top_10_data['DefectRate'].min()], color='lightgreen')
    axes[0, 1].barh(['Top 10 (Peor)'], [top_10_data['DefectRate'].max()], color='lightyellow')
    axes[0, 1].set_xlabel("Tasa de Defectos (%)")
    axes[0, 1].set_title("Tasa de Defectos")

    # Plot 3: Productividad
    axes[0, 2].barh(['Promedio'], [avg_productivity], color='lightcoral')
    axes[0, 2].barh(['Top 10 (Mejor)'], [top_10_data['WorkerProductivity'].max()], color='lightgreen')
    axes[0, 2].barh(['Top 10 (Peor)'], [top_10_data['WorkerProductivity'].min()], color='lightyellow')
    axes[0, 2].set_xlabel("Productividad (%)")
    axes[0, 2].set_title("Productividad del Trabajador")

    # Plot 4: Energía
    axes[1, 0].barh(['Promedio'], [avg_energy], color='lightcoral')
    axes[1, 0].barh(['Top 10 (Mejor)'], [top_10_data['EnergyConsumption'].min()], color='lightgreen')
    axes[1, 0].barh(['Top 10 (Peor)'], [top_10_data['EnergyConsumption'].max()], color='lightyellow')
    axes[1, 0].set_xlabel("Consumo Energético ($)")
    axes[1, 0].set_title("Consumo de Energía")

    # Plot 5: Downtime
    axes[1, 1].barh(['Promedio'], [avg_downtime], color='lightcoral')
    axes[1, 1].barh(['Top 10 (Mejor)'], [top_10_data['DowntimePercentage'].min()], color='lightgreen')
    axes[1, 1].barh(['Top 10 (Peor)'], [top_10_data['DowntimePercentage'].max()], color='lightyellow')
    axes[1, 1].set_xlabel("Porcentaje (%)")
    axes[1, 1].set_title("Downtime")

    # Plot 6: Calidad
    axes[1, 2].barh(['Promedio'], [avg_quality], color='lightcoral')
    axes[1, 2].barh(['Top 10 (Mejor)'], [top_10_data['QualityScore'].max()], color='lightgreen')
    axes[1, 2].barh(['Top 10 (Peor)'], [top_10_data['QualityScore'].min()], color='lightyellow')
    axes[1, 2].set_xlabel("Score (0-100)")
    axes[1, 2].set_title("Quality Score")

    plt.tight_layout()
    st.pyplot(comparison_fig)

    # ESTADÍSTICAS RESUMEN
    st.markdown("---")
    st.subheader("Estadísticas Resumen del Top 10")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            "Ahorro Promedio (vs promedio)",
            f"${(avg_cost + avg_energy + avg_material) - (top_10_data['ProductionCost'] + top_10_data['EnergyConsumption'] + top_10_data['AdditiveMaterialCost']).mean():,.0f}",
            "por lote"
        )

    with summary_col2:
        st.metric(
            "Reducción de Defectos",
            f"{((avg_defect - top_10_data['DefectRate'].mean()) / avg_defect * 100):.1f}%",
            f"de {avg_defect:.2f}% a {top_10_data['DefectRate'].mean():.2f}%"
        )

    with summary_col3:
        st.metric(
            "Mejora en Productividad",
            f"{((top_10_data['WorkerProductivity'].mean() - avg_productivity) / avg_productivity * 100):.1f}%",
            f"promedio top 10"
        )


# ============================================================================
# SECCIÓN D: PREDICCIÓN DE CAMBIOS CON REGRESIÓN
# ============================================================================
elif section == "🔮 D. Predicción de Cambios":
    st.title("🔮 D. Predicción de Cambios - Modelo de Regresión")

    st.markdown("""
    Usa un **modelo de regresión entrenado con tus datos** para predecir
    cómo cambios en 3 variables clave afectan el Score Global:

    **Variables que puedes ajustar:**
    - Horas de Mantenimiento
    - Tasa de Defectos
    - Quality Score
    """)

    st.markdown("---")

    # ENTRENAR MODELO DE REGRESIÓN
    st.subheader("Entrenando Modelo de Regresión...")

    # Preparar datos para el modelo
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_norm_model = pd.DataFrame(
        MinMaxScaler(feature_range=(0, 100)).fit_transform(df[numeric_cols]),
        columns=numeric_cols
    )

    # Calcular Score Global para cada escenario del dataset
    scores_full = pd.DataFrame({
        'Costo': 100 - df_norm_model['ProductionCost'] * 0.4 - df_norm_model['EnergyConsumption'] * 0.35 - df_norm_model['AdditiveMaterialCost'] * 0.25,
        'Eficiencia': df_norm_model['WorkerProductivity'] * 0.4 + df_norm_model['EnergyEfficiency'] * 0.3 + df_norm_model['InventoryTurnover'] * 0.3,
        'Calidad': (100 - df_norm_model['DefectRate']) * 0.5 + df_norm_model['QualityScore'] * 0.5
    })
    scores_full['ScoreGlobal'] = scores_full['Costo'] * 0.35 + scores_full['Eficiencia'] * 0.35 + scores_full['Calidad'] * 0.30

    # Variables predictoras (las 3 clave)
    X = df[['MaintenanceHours', 'DefectRate', 'QualityScore']].values
    y = scores_full['ScoreGlobal'].values

    # Entrenar modelo (Random Forest es mejor para relaciones no-lineales)
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)

    st.success("Modelo entrenado correctamente")

    st.markdown("---")
    st.subheader("Ajusta los Parámetros")

    col1, col2, col3 = st.columns(3)

    with col1:
        maintenance = st.slider(
            "Horas de Mantenimiento",
            min_value=0.0,
            max_value=23.0,
            value=df['MaintenanceHours'].mean(),
            step=0.5,
            help="Horas de mantenimiento por período"
        )

    with col2:
        defect_rate = st.slider(
            "Tasa de Defectos (%)",
            min_value=0.1,
            max_value=5.0,
            value=df['DefectRate'].mean(),
            step=0.1,
            help="Porcentaje de defectos"
        )

    with col3:
        quality_score = st.slider(
            "Quality Score",
            min_value=60.0,
            max_value=100.0,
            value=df['QualityScore'].mean(),
            step=1.0,
            help="Score de calidad (0-100)"
        )

    st.markdown("---")

    # REALIZAR PREDICCIÓN
    st.subheader("Predicción del Score Global")

    input_data = np.array([[maintenance, defect_rate, quality_score]])
    predicted_score = model.predict(input_data)[0]

    # Comparar con promedio del dataset
    avg_score = y.mean()
    score_diff = predicted_score - avg_score

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Score Global Predicho",
            f"{predicted_score:.1f}/100",
            f"{score_diff:+.1f} vs promedio"
        )

    with col2:
        st.metric(
            "Promedio Dataset",
            f"{avg_score:.1f}/100"
        )

    with col3:
        if score_diff > 5:
            st.success(f"Excelente (+{score_diff:.1f})")
        elif score_diff > 0:
            st.info(f"Bueno (+{score_diff:.1f})")
        else:
            st.warning(f"Por debajo promedio ({score_diff:.1f})")

    st.markdown("---")

    # ANÁLISIS DETALLADO
    st.subheader("Detalles de la Predicción")

    explain_col1, explain_col2 = st.columns(2)

    with explain_col1:
        st.write("**Parámetros Seleccionados:**")
        st.write(f"  • Mantenimiento: {maintenance:.1f}h ({((maintenance - df['MaintenanceHours'].mean()) / df['MaintenanceHours'].mean() * 100):+.1f}% vs promedio)")
        st.write(f"  • Defectos: {defect_rate:.2f}% ({((defect_rate - df['DefectRate'].mean()) / df['DefectRate'].mean() * 100):+.1f}% vs promedio)")
        st.write(f"  • Quality Score: {quality_score:.1f}/100 ({((quality_score - df['QualityScore'].mean()) / df['QualityScore'].mean() * 100):+.1f}% vs promedio)")

    with explain_col2:
        st.write("**Interpretación del Score:**")
        st.write(f"  • {predicted_score:.1f}/100 significa que tu configuración está")
        st.write(f"    en el percentil {(predicted_score / 100 * 100):.0f}% de todas las opciones")
        st.write(f"  • Score > 75: Excelente balance")
        st.write(f"  • Score 50-75: Balance aceptable")
        st.write(f"  • Score < 50: Balance deficiente")

    st.markdown("---")

    # COMPARACIÓN CON DATASET
    st.subheader("Visualización: Tu Configuración vs Dataset")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Distribución de scores
    ax.hist(y, bins=40, color='lightblue', edgecolor='black', alpha=0.7, label='Dataset')
    ax.axvline(avg_score, color='blue', linestyle='--', linewidth=2, label=f'Promedio: {avg_score:.1f}')
    ax.axvline(predicted_score, color='red', linestyle='--', linewidth=2, label=f'Tu config: {predicted_score:.1f}')
    ax.set_xlabel('Score Global')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Scores en Dataset vs Tu Configuración')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # TABLA DE COMPARACIÓN
    st.subheader("Tabla de Escenarios Similares")

    # Encontrar escenarios en dataset similares a la predicción
    df_comparison = pd.DataFrame({
        'MaintenanceHours': df['MaintenanceHours'],
        'DefectRate': df['DefectRate'],
        'QualityScore': df['QualityScore'],
        'ScoreGlobal': scores_full['ScoreGlobal']
    })

    # Calcular distancia euclidiana respecto a la configuración actual
    distances = np.sqrt(
        (df_comparison['MaintenanceHours'] - maintenance)**2 +
        (df_comparison['DefectRate'] - defect_rate)**2 +
        (df_comparison['QualityScore'] - quality_score)**2
    )

    df_comparison['Distancia'] = distances
    similar = df_comparison.nsmallest(5, 'Distancia')

    st.write("**5 Escenarios más similares en el dataset:**")
    st.dataframe(
        similar[['MaintenanceHours', 'DefectRate', 'QualityScore', 'ScoreGlobal', 'Distancia']].reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")

    # INFORMACIÓN DEL MODELO
    with st.expander("Información Técnica del Modelo"):
        st.write("""
        **Tipo de Modelo:** Random Forest Regressor

        **Datos de Entrenamiento:**
        - Registros usados: 3,240 escenarios
        - Variables predictoras: MaintenanceHours, DefectRate, QualityScore
        - Variable objetivo: Score Global (0-100)

        **Características del Modelo:**
        - 100 árboles de decisión
        - Profundidad máxima: 10 niveles
        - Captura relaciones no-lineales

        **Ventajas:**
        - Aprende patrones del dataset real
        - No requiere asumir relaciones lineales
        - Honesto: si la correlación es débil, el modelo lo refleja

        **Limitaciones:**
        - Solo predice con los 3 parámetros seleccionados
        - No puede extrapolar fuera del rango del dataset
        - La precisión depende de la calidad de los datos de entrenamiento
        """)
