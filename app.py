import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
from utils import plot_regresion_modelo# para usar funciones auxiliares

# ─────────────────────────────────────────────
# 🏎️ Portada visual
# ─────────────────────────────────────────────
st.image("images/portada.jpeg", use_container_width=True)
st.title("Predicción de rendimiento en Fórmula 1")

# ─────────────────────────────────────────────
# 🏎️ Cargar modelo y dataset
# ─────────────────────────────────────────────
try:
    modelo = joblib.load("modelo/modelo_random_forest_enriquecido.pkl")
    st.success("✅ Modelo cargado: Random Forest enriquecido")
except Exception as e:
    st.error(f"⚠️ Error al cargar el modelo: {e}")
    st.stop()

try:
    df_validacion = pd.read_csv("data/df_validacion.csv")
except FileNotFoundError:
    st.error("❌ No se encontró df_validacion.csv en /data.")
    st.stop()

df_resultados = df_validacion[["LapTime_real", "LapTime_predicho"]].copy()

# ─────────────────────────────────────────────
# 🔀 Selector de sección
# ─────────────────────────────────────────────
seccion = st.radio("Seleccioná una sección", [
    "🏎️ Entrenamiento y evaluación",
    "🏁 Validación y visualización"
])

# ─────────────────────────────────────────────
# 🏎️ Sección 1: Entrenamiento y evaluación
# ─────────────────────────────────────────────
if seccion == "🏎️ Entrenamiento y evaluación":
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏎️ Dataset y preprocesamiento",
        "🏎️ Modelos base",
        "🏎️ GridSearch",
        "🏎️ Validación RL",
        "🏎️ Features",
        "🏁 Modelo beneficiado"
    ])

    # ── TAB 0 ──
    with tab0:
        st.subheader("🏎️ Dataset y preprocesamiento")
        st.dataframe(df_validacion.head())
        st.markdown("""
        - División del conjunto con `GroupKFold(n_splits=5)` para respetar eventos.
        - Preprocesamiento con `ColumnTransformer`:
            - Pipeline numérico: `StandardScaler`
            - Pipeline categórico: `OneHotEncoder`
        - Integración en pipeline completo con `make_pipeline(modelo)`
        """)

    # ── TAB 1 ──
    with tab1:
        st.subheader("🏎️ Comparación de modelos base")
        df_metricas_modelos = pd.DataFrame({
            "Modelo": ["Linear Regression", "Gradient Boosting", "Random Forest"],
            "MAE": [6.64, 14.14, 14.69],
            "RMSE": [9.21, 18.47, 19.16],
            "R²": [0.67, -0.32, -0.42]
        })
        st.dataframe(df_metricas_modelos)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].bar(df_metricas_modelos["Modelo"], df_metricas_modelos["MAE"], color="orange")
        axs[0].set_title("MAE por modelo")
        axs[1].bar(df_metricas_modelos["Modelo"], df_metricas_modelos["RMSE"], color="red")
        axs[1].set_title("RMSE por modelo")
        axs[2].bar(df_metricas_modelos["Modelo"], df_metricas_modelos["R²"], color="green")
        axs[2].set_title("R² por modelo")
        st.pyplot(fig)
 # ─────────────────────────────────────────────
        # 🏎️ Gráfico de dispersión — Regresión Lineal
        # ─────────────────────────────────────────────
        st.subheader("🏎️ Regresión Lineal: y_test vs y_pred")

        try:
            # 🔹 Cargar datos de test y predicción lineal
            y_test_lr = pd.read_csv("data/y_test_lr.csv")["LapTime_real"]
            y_pred_lr = pd.read_csv("data/y_pred_lr.csv")["LapTime_predicho"]

            # 🔹 Mostrar gráfico
            st.altair_chart(plot_regresion_modelo(y_test_lr, y_pred_lr, "Regresión Lineal"), use_container_width=True)

        except FileNotFoundError:
            st.warning("⚠️ No se encontraron los archivos 'y_test_lr.csv' o 'y_pred_lr.csv'. Verificá la carpeta 'data'.")
    # ── TAB 2 ──
    with tab2:
        st.subheader("🏎️ Ajuste de hiperparámetros para Random Forest")
        df_ajuste = pd.read_csv("data/ajuste_rf.csv")
        st.dataframe(df_ajuste)
        st.markdown("""
        Se aplicó `GridSearchCV` con los hiperparámetros más relevantes.
        El modelo optimizado logra mejor balance entre sesgo y varianza.
        """)

    # ── TAB 3 ──
    with tab3:
        st.subheader("🏎️ Validación cruzada: Regresión Lineal")
        df_folds = pd.read_csv("data/cv_rl.csv")
        st.dataframe(df_folds)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_folds["Fold"], df_folds["R²"], color="orange")
        ax.axhline(df_folds["R² medio"].iloc[0], color="red", linestyle="--", label="R² medio")
        ax.legend()
        st.pyplot(fig)

    # ── TAB 4 ──
    with tab4:
        st.subheader("🏎️ Nuevas features")

        st.markdown("""
    Se incorporaron dos nuevas variables al modelo para mejorar la capacidad predictiva:

    - `Diff_Speed`: diferencia entre la velocidad promedio en sector técnico (`SpeedST`) y la velocidad en vuelta rápida (`SpeedFL`).
    - `WeatherIndex`: índice climático calculado como la temperatura de pista multiplicada por un factor de lluvia.

    ```python
    df_modelo["Diff_Speed"] = df_modelo["SpeedST"] - df_modelo["SpeedFL"]
    df_modelo["WeatherIndex"] = df_modelo["TrackTemp"] * (1 + df_modelo["Rainfall"].astype(int)*0.1)
    ```
    """)

        st.subheader("🏎️ Ingeniería ligera de features")
        df_resultados_ext = pd.read_csv("data/resultados_ext.csv")
        st.dataframe(df_resultados_ext)

    # ── TAB 5 ──
    with tab5:
        st.subheader("🏎️ Modelo beneficiado: Random Forest enriquecido")
        df_errores = pd.read_csv("data/errores_modelos.csv")
        errores_rf = df_errores["error_rf"]
        errores_lr = df_errores["error_lr"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        axes[0].hist(errores_rf, bins=30, alpha=0.7, color='steelblue')
        axes[0].set_title("Error Random Forest")
        axes[1].hist(errores_lr, bins=30, alpha=0.7, color='orange')
        axes[1].set_title("Error Linear Regression")
        st.pyplot(fig)
    # ─────────────────────────────────────────────
        # 📈 Gráfico de dispersión — Random Forest enriquecido
        # ─────────────────────────────────────────────
        st.subheader("📈 Predicciones vs Reales - Random Forest enriquecido")

        try:
            # 🔹 Cargar datos de test y predicción
            y_test_rf = pd.read_csv("data/y_test_rf.csv")["LapTime_real"]
            y_pred_rf = pd.read_csv("data/y_pred_rf.csv")["LapTime_predicho"]

            # 🔹 Mostrar gráfico
            st.altair_chart(plot_regresion_modelo(y_test_rf, y_pred_rf, "Random Forest enriquecido"), use_container_width=True)

        except FileNotFoundError:
            st.warning("⚠️ No se encontraron los archivos 'y_test_rf.csv' o 'y_pred_rf.csv'. Verificá la carpeta 'data'.")

#
# ─────────────────────────────────────────────
# 🏁 Sección 2: Validación y visualización
# ─────────────────────────────────────────────
elif seccion == "🏁 Validación y visualización":
    tab6, tab7, tab8 = st.tabs([
        "🏎️ Hipótesis",
        "🏎️ Altair para Hipótesis",
        "🏎️ Altair para errores"
    ])

    # ─────────────────────────────────────────────
    # 🏎️ TAB 6 — Visualización de hipótesis
    # ─────────────────────────────────────────────
    with tab6:
        st.subheader("🏎️ Validación con hipótesis")

    # 📜 Hipótesis de validación
        st.markdown("### 🏎️  Hipótesis principal")
        st.markdown("""
    > **Si el modelo predice bien, entonces debería identificar correctamente la mejor vuelta en cada circuito.**
    """)

        st.markdown("### 🏎️  Hipótesis derivada")
        st.markdown("""
    > **Si el modelo captura correctamente el rendimiento relativo de los pilotos, entonces el verdadero ganador debería estar entre los tres con mejor tiempo predicho en cada circuito.**
    """)

        try:
            df_comparacion = pd.read_csv("data/comparacion_vueltas_rf.csv")
            df_hipotesis = pd.read_csv("data/hipotesis_rf.csv")
            df_hipotesis_derivada = pd.read_csv("data/hipotesis_derivada_rf.csv")

            st.success("✅ Archivos cargados correctamente desde /data")
            st.dataframe(df_comparacion)
            st.dataframe(df_hipotesis)
            st.dataframe(df_hipotesis_derivada)
        except FileNotFoundError:
            st.error("⚠️ Faltan archivos CSV en la carpeta /data. Verificá que existan los tres datasets.")

    # ─────────────────────────────────────────────
    # 🏎️ TAB 7 — Validación de la Hipótesis (Altair)
    # ─────────────────────────────────────────────
    with tab7:
        st.header("🏎️ Validación de la Hipótesis — Comparación Real vs Predicho por Continente")

        try:
            df = pd.read_csv("data/validacion_hipotesis.csv")
            st.success(f"✅ Archivo cargado correctamente con {len(df)} filas desde /data.")
        except FileNotFoundError:
            st.error("⚠️ No se encontró 'data/validacion_hipotesis.csv'. Verificá la ruta y el nombre del archivo.")
            st.stop()

        mapa_continente = {
            "Spanish Grand Prix": "Europa", "British Grand Prix": "Europa", "Monaco Grand Prix": "Europa",
            "Austrian Grand Prix": "Europa", "Hungarian Grand Prix": "Europa", "Dutch Grand Prix": "Europa",
            "Belgian Grand Prix": "Europa", "Italian Grand Prix": "Europa", "Emilia Romagna Grand Prix": "Europa",
            "French Grand Prix": "Europa", "Portuguese Grand Prix": "Europa", "Turkish Grand Prix": "Europa",
            "Tuscan Grand Prix": "Europa", "Eifel Grand Prix": "Europa", "70th Anniversary Grand Prix": "Europa",
            "Styrian Grand Prix": "Europa",
            "Brazilian Grand Prix": "América", "Mexico City Grand Prix": "América", "United States Grand Prix": "América",
            "Miami Grand Prix": "América", "Las Vegas Grand Prix": "América", "Canadian Grand Prix": "América",
            "São Paulo Grand Prix": "América",
            "Japanese Grand Prix": "Asia", "Singapore Grand Prix": "Asia", "Azerbaijan Grand Prix": "Asia",
            "Qatar Grand Prix": "Asia", "Saudi Arabian Grand Prix": "Asia", "Bahrain Grand Prix": "Asia",
            "Abu Dhabi Grand Prix": "Asia",
            "Australian Grand Prix": "Oceanía",
            "Pre-Season Test": "Desconocido", "Pre-Season Test 1": "Desconocido",
            "Pre-Season Test 2": "Desconocido", "Pre-Season Testing": "Desconocido"
        }

        tipo_circuito = {
            'Monaco Grand Prix': 'callejero', 'Azerbaijan Grand Prix': 'callejero', 'Singapore Grand Prix': 'callejero',
            'Miami Grand Prix': 'callejero', 'Las Vegas Grand Prix': 'callejero', 'Saudi Arabian Grand Prix': 'callejero',
            'Australian Grand Prix': 'permanente', 'Austrian Grand Prix': 'permanente', 'Bahrain Grand Prix': 'permanente',
            'Belgian Grand Prix': 'permanente', 'British Grand Prix': 'permanente', 'Canadian Grand Prix': 'permanente',
            'Dutch Grand Prix': 'permanente', 'Emilia Romagna Grand Prix': 'permanente', 'French Grand Prix': 'permanente',
            'Hungarian Grand Prix': 'permanente', 'Italian Grand Prix': 'permanente', 'Japanese Grand Prix': 'permanente',
            'Mexico City Grand Prix': 'permanente', 'Portuguese Grand Prix': 'permanente', 'Qatar Grand Prix': 'permanente',
            'Russian Grand Prix': 'permanente', 'Spanish Grand Prix': 'permanente', 'Styrian Grand Prix': 'permanente',
            'São Paulo Grand Prix': 'permanente', 'Turkish Grand Prix': 'permanente', 'Tuscan Grand Prix': 'permanente',
            'United States Grand Prix': 'permanente', 'Eifel Grand Prix': 'permanente', 'Sakhir Grand Prix': 'permanente',
            '70th Anniversary Grand Prix': 'permanente',
            'Pre-Season Test': 'test', 'Pre-Season Test 1': 'test', 'Pre-Season Test 2': 'test', 'Pre-Season Testing': 'test'
        }

        df["Continente"] = df.get("Continente", df["EventName"].map(mapa_continente)).fillna("Desconocido")
        df["tipo_circuito"] = df.get("tipo_circuito", df["EventName"].map(tipo_circuito)).fillna("desconocido")

        if "Resultado" not in df.columns:
            df["Resultado"] = df.apply(
                lambda x: "ACIERTO" if x["Driver_real"] == x.get("Driver_predicho", None) else "ERROR",
                axis=1
            )

        columnas_necesarias = [
            "EventName", "Driver_real", "LapTime_real",
            "Driver_predicho", "LapTime_predicho",
            "Team", "longitud_km", "Compound", "Rainfall", "Continente", "tipo_circuito"
        ]
        faltantes = [c for c in columnas_necesarias if c not in df.columns]
        if faltantes:
            st.error(f"⚠️ Faltan columnas en el dataset: {faltantes}")
            st.stop()

        continentes = sorted(df["Continente"].dropna().unique().tolist())
        if not continentes:
            st.error("❌ No se detectaron continentes para mostrar.")
            st.stop()

        tabs_continentes = st.tabs([f"🌍 {c}" for c in continentes])

        for tab_obj, cont in zip(tabs_continentes, continentes):
            with tab_obj:
                st.subheader(f"🏁 Resultados — {cont}")
                df_cont = df[df["Continente"] == cont].copy()

                if df_cont.empty:
                    st.info("ℹ️ No hay registros para este continente.")
                    continue

                             # === 7️⃣ SELECCIÓN INTERACTIVA ===
                seleccion = alt.selection_single(fields=["EventName"], empty="none", clear="true", on="click")

                # === 8️⃣ GRÁFICO MAESTRO ===
                maestro = (
                    alt.Chart(df_cont)
                    .mark_circle(size=200, opacity=0.9, stroke="black", strokeWidth=0.3)
                    .encode(
                        x=alt.X("LapTime_real:Q", title="🏎️ Mejor vuelta real (s)"),
                        y=alt.Y("LapTime_predicho:Q", title="🏎️ Vuelta predicha (s)"),
                        color=alt.Color(
                            "Resultado:N",
                            title="Resultado",
                            scale=alt.Scale(domain=["ACIERTO", "ERROR"], range=["#00C853", "#D50000"])
                        ),
                        tooltip=[
                            alt.Tooltip("EventName:N", title="Circuito"),
                            alt.Tooltip("tipo_circuito:N", title="Tipo de circuito"),
                            alt.Tooltip("Driver_real:N", title="Piloto real"),
                            alt.Tooltip("LapTime_real:Q", title="Tiempo real (s)", format=".3f"),
                            alt.Tooltip("Driver_predicho:N", title="Piloto predicho"),
                            alt.Tooltip("LapTime_predicho:Q", title="Tiempo predicho (s)", format=".3f"),
                            alt.Tooltip("Team:N", title="Equipo"),
                            alt.Tooltip("Compound:N", title="Compuesto"),
                            alt.Tooltip("Rainfall:Q", title="Lluvia (mm)")
                        ]
                    )
                    .add_params(seleccion)
                    .properties(width=700, height=380, title=f"🏎️ Comparación Real vs Predicho — {cont}")
                )

                # === 9️⃣ GRÁFICO DETALLE — velocímetro
                detalle_df = df_cont.copy()
                detalle_df["longitud_km"] = pd.to_numeric(detalle_df["longitud_km"], errors="coerce")
                detalle_df["info_circuito"] = (
                    detalle_df["EventName"].astype(str)
                    + " | " + detalle_df["Team"].astype(str)
                    + " | " + detalle_df["Compound"].astype(str)
                    + " | Lluvia: " + detalle_df["Rainfall"].astype(str) + " mm"
                )

                detalle = (
                    alt.Chart(detalle_df)
                    .transform_filter(seleccion)
                    .transform_calculate(potencia="datum.longitud_km / 10")
                )

                fondo = (
                    alt.Chart(pd.DataFrame({'angle': [0], 'radius': [1]}))
                    .mark_arc(innerRadius=60, outerRadius=100, color="#222")
                    .encode()
                    .properties(width=350, height=250)
                )

                arco = (
                    detalle.mark_arc(innerRadius=60, outerRadius=100)
                    .encode(
                        theta=alt.Theta("potencia:Q", stack=False, scale=alt.Scale(domain=[0,1], range=[0,270])),
                        color=alt.value("#1E88E5")
                    )
                )

                aguja = (
                    detalle.mark_point(filled=True, color="#FF1744", size=200)
                    .encode(
                        x=alt.X("cos(potencia * PI() * 1.5):Q", title=None),
                        y=alt.Y("sin(potencia * PI() * 1.5):Q", title=None)
                    )
                )

                texto = (
                    alt.Chart(detalle_df)
                    .transform_filter(seleccion)
                    .mark_text(align='center', baseline='middle', size=13, color='white', lineBreak='\n')
                    .encode(text="info_circuito:N")
                    .properties(width=350, height=250)
                )

                velocimetro = (
                    fondo + arco + aguja + texto
                ).properties(title="⚙️ Detalle técnico — Potencia del circuito (velocímetro)")

                # === 🔟 COMPOSICIÓN FINAL
                final = alt.vconcat(maestro, velocimetro).resolve_legend(color="independent")
                st.altair_chart(final, use_container_width=True)
                # ─────────────────────────────────────────────
        # 🏎️ NUEVO GRÁFICO: Relación entre velocidad y tiempo de vuelta
        # ─────────────────────────────────────────────
        st.subheader("🏎️ Relación entre velocidad máxima y tiempo de vuelta")

        try:
            # Cargar dataset reducido desde /data
            df_graficos = pd.read_csv("data/df_graficos.csv")
            st.success(f"✅ Dataset para gráfico Altair cargado correctamente ({len(df_graficos)} filas)")
        except FileNotFoundError:
            st.error("⚠️ No se encontró el archivo 'data/df_graficos.csv'. Verificá la carpeta /data.")
            st.stop()

        # Crear gráfico Altair
        import altair as alt

        chart_corr = (
            alt.Chart(df_graficos)
            .mark_circle(size=90, opacity=0.8, stroke="black", strokeWidth=0.3)
            .encode(
                x=alt.X('SpeedST:Q', title='Velocidad máxima (km/h)'),
                y=alt.Y('LapTime:Q', title='Tiempo de vuelta (s)'),
                color=alt.Color('tipo_circuito:N', title='Tipo de circuito',
                                scale=alt.Scale(scheme='set2')),
                size=alt.Size('TrackTemp:Q', title='Temperatura de pista (°C)', legend=None),
                tooltip=[
                    alt.Tooltip('DriverName:N', title='Piloto'),
                    alt.Tooltip('Team:N', title='Equipo'),
                    alt.Tooltip('EventName:N', title='Circuito'),
                    alt.Tooltip('SpeedST:Q', title='Velocidad (km/h)', format='.1f'),
                    alt.Tooltip('LapTime:Q', title='Tiempo (s)', format='.3f'),
                    alt.Tooltip('TrackTemp:Q', title='Temp. pista (°C)', format='.1f')
                ]
            )
            .properties(
                title='🏁 Relación entre velocidad punta y tiempo de vuelta — Temporada 2023',
                width=750,
                height=450
            )
            .interactive()
        )

        # Mostrar gráfico en Streamlit
        st.altair_chart(chart_corr, use_container_width=True)

        # Agregar breve interpretación debajo
        st.markdown("""
        **🏎️ Interpretación:**
        - Cada punto representa una vuelta de un piloto.  
        - Cuanto **mayor la velocidad**, menor suele ser el **tiempo de vuelta** (relación inversa esperada).  
        - Los colores diferencian el **tipo de circuito** (callejero o permanente).  
        - El tamaño indica la **temperatura de pista**, afectando el rendimiento y desgaste de neumáticos.
        """)
           # ─────────────────────────────────────────────
    # 🏎️ TAB 8 — Comparación interactiva con Altair
    # ─────────────────────────────────────────────
    with tab8:
        st.subheader("🏎️ Comparación interactiva — LapTime real vs predicho con error")

        try:
            df_validacion_hipotesis = pd.read_csv("data/validacion_hipotesis.csv")
            st.success(f"✅ Archivo cargado correctamente con {len(df_validacion_hipotesis)} filas.")
        except FileNotFoundError:
            st.error("⚠️ No se encontró 'data/validacion_hipotesis.csv'. Verificá la ruta y el nombre del archivo.")
            st.stop()

        # 🔹 1. Calcular error absoluto
        df_validacion_hipotesis["Error_abs"] = np.abs(
            df_validacion_hipotesis["LapTime_real"] - df_validacion_hipotesis["LapTime_predicho"]
        )

        # 🔹 2. Gráfico Altair interactivo
        chart = (
            alt.Chart(df_validacion_hipotesis)
            .mark_circle(size=80, opacity=0.9, stroke="black", strokeWidth=0.6)
            .encode(
                x=alt.X("LapTime_real:Q", title="🏎️ LapTime real (s)", scale=alt.Scale(domain=[65, 115])),
                y=alt.Y("LapTime_predicho:Q", title="🏎️ LapTime predicho (s)", scale=alt.Scale(domain=[65, 115])),
                color=alt.Color("EventName:N", title="Evento", scale=alt.Scale(scheme="tableau10")),
                tooltip=[
                    alt.Tooltip("EventName:N", title="Evento"),
                    alt.Tooltip("Driver_real:N", title="Piloto real"),
                    alt.Tooltip("Driver_predicho:N", title="Piloto predicho"),
                    alt.Tooltip("LapTime_real:Q", title="Tiempo real (s)", format=".3f"),
                    alt.Tooltip("LapTime_predicho:Q", title="Tiempo predicho (s)", format=".3f"),
                    alt.Tooltip("Error_abs:Q", title="Error absoluto (s)", format=".3f")
                ]
            )
            .properties(
                width=750,
                height=450,
                title="🏁 Comparación: LapTime real vs predicho por evento (con error)"
            )
        )

        # 🔹 3. Línea ideal (y = x)
        min_val = min(
            df_validacion_hipotesis["LapTime_real"].min(),
            df_validacion_hipotesis["LapTime_predicho"].min()
        )
        max_val = max(
            df_validacion_hipotesis["LapTime_real"].max(),
            df_validacion_hipotesis["LapTime_predicho"].max()
        )
        linea_ideal = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})

        linea = (
            alt.Chart(linea_ideal)
            .mark_line(strokeDash=[5, 5], color="red")
            .encode(x="x:Q", y="y:Q")
        )

        # 🔹 4. Composición final
        final_chart = chart + linea
        st.altair_chart(final_chart, use_container_width=True)

        # 🔹 5. Ranking de errores
        columnas_ranking = [
            "EventName", "Driver_real", "Driver_predicho",
            "LapTime_real", "LapTime_predicho", "Error_abs"
        ]
        columnas_disponibles = [c for c in columnas_ranking if c in df_validacion_hipotesis.columns]

        if columnas_disponibles:
            df_error_rank = df_validacion_hipotesis.sort_values("Error_abs", ascending=False)
            st.markdown("### 🏎️ Ranking de errores por evento (de mayor a menor diferencia)")
            st.dataframe(df_error_rank[columnas_disponibles])
        else:
            st.warning("⚠️ No se encontraron las columnas necesarias para mostrar el ranking.")

        # 🔹 6. Estadísticas globales
        if "LapTime_real" in df_validacion_hipotesis.columns and "LapTime_predicho" in df_validacion_hipotesis.columns:
            error_global = df_validacion_hipotesis["LapTime_real"] - df_validacion_hipotesis["LapTime_predicho"]
            st.markdown("### 🏎️ Estadísticas globales del error (LapTime_real - LapTime_predicho)")
            st.write(f"Media del error: {error_global.mean():.3f} s")
            st.write(f"Mediana del error: {error_global.median():.3f} s")
            st.write(f"Error mínimo: {error_global.min():.3f} s")
            st.write(f"Error máximo: {error_global.max():.3f} s")
            st.write(f"Desviación estándar: {error_global.std():.3f} s")
        else:
            st.warning("⚠️ No se pueden calcular estadísticas globales: faltan columnas de LapTime.")
