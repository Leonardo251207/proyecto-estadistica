import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from scipy import stats
import google.generativeai as genai

# Configuración de la página
st.set_page_config(page_title="Análisis Estadístico Profesional", layout="wide")

# Título de la App (Sección 1 del reporte)
st.title("📊 Aplicación de Análisis Estadístico y Pruebas de Hipótesis")
st.markdown("---")

# --- 1. MÓDULO DE CARGA DE DATOS ---
st.header("1. Carga de Datos")
tipo_datos = st.radio("Selecciona origen de datos:", ["Generación sintética", "Cargar CSV"])

df = None

if tipo_datos == "Generación sintética":
    # Generamos datos con una distribución normal para pruebas controladas
    datos = np.random.normal(loc=50, scale=10, size=100)
    df = pd.DataFrame(datos, columns=["Variable_X"])
    st.success("✅ Datos sintéticos generados correctamente (n=100)")
else:
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.success("✅ Archivo cargado con éxito")

# Solo ejecutamos si hay datos cargados
if df is not None:
    st.markdown("---")
    st.write("### Vista previa de los datos")
    st.dataframe(df.head(), use_container_width=True)

   # --- 2. DESCRIPCIÓN DE LOS DATOS (DISEÑO FINAL Y LIMPIO) ---
    st.header("2. Descripción de los Datos")
    
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columnas_numericas) > 0:
        col_analisis = st.selectbox("Selecciona la variable numérica para analizar:", columnas_numericas)
        
        # Métricas rápidas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Media", f"{df[col_analisis].mean():.2f}")
        m2.metric("Mediana", f"{df[col_analisis].median():.2f}")
        m3.metric("Desv. Est.", f"{df[col_analisis].std():.2f}")
        m4.metric("Tamaño (n)", len(df[col_analisis]))
        
        st.markdown("---")

        col_grafica, col_preguntas = st.columns([2, 1])
        
        with col_grafica:
            st.write("### 📈 Visualización Dinámica")
            tab_hist, tab_box = st.tabs(["Histograma", "Boxplot"])
            
            with tab_hist:
                plt.style.use('ggplot') 
                fig_h, ax_h = plt.subplots(figsize=(10, 5))
                sns.histplot(df[col_analisis], kde=True, ax=ax_h, color="#1f77b4", edgecolor="white")
                
                # TITULO GENÉRICO Y SIN ETIQUETAS DUPLICADAS
                ax_h.set_title("Distribución de los Datos", fontweight='bold', fontsize=12)
                ax_h.set_ylabel("") # Quitamos "count"
                ax_h.set_xlabel("") # QUITAMOS EL ETIQUETADO QUE SE CORTABA
                
                plt.tight_layout()
                st.pyplot(fig_h)
                
                # AQUÍ VA LA PREGUNTA COMPLETA (LEGIBLE Y SIN CORTAR)
                st.caption(f"**Variable analizada:** {col_analisis}")
                
                buf_h = io.BytesIO()
                fig_h.savefig(buf_h, format="png", bbox_inches='tight')
                st.download_button("💾 Descargar Histograma", buf_h.getvalue(), "histograma.png", "image/png", key="btn_hist")

            with tab_box:
                fig_b, ax_b = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=df[col_analisis], ax=ax_b, color="#ff7f0e")
                
                ax_b.set_title("Diagrama de Caja (Outliers)", fontweight='bold', fontsize=12)
                ax_b.set_xlabel("") # QUITAMOS EL ETIQUETADO
                
                plt.tight_layout()
                st.pyplot(fig_b)
                
                st.caption(f"**Variable analizada:** {col_analisis}")
                
                buf_b = io.BytesIO()
                fig_b.savefig(buf_b, format="png", bbox_inches='tight')
                st.download_button("💾 Descargar Boxplot", buf_b.getvalue(), "boxplot.png", "image/png", key="btn_box")

        with col_preguntas:
            st.write("### 📝 Análisis")
            st.radio("¿Distribución normal?", ["Sí", "No"], key="obs_norm")
            st.radio("¿Sesgo?", ["No", "Izquierda", "Derecha"], key="obs_sesgo")
            st.radio("¿Hay Outliers?", ["No", "Sí"], key="obs_out")
            st.dataframe(df[col_analisis].describe().to_frame().T.round(2), use_container_width=True)
            
    else:
        st.error("❌ No se encontraron columnas numéricas.")

    # --- 3. PLANTEAMIENTO DE LA PRUEBA DE HIPÓTESIS (AVANCE) ---
    st.markdown("---")
    st.header("3. Planteamiento de la Prueba de Hipótesis")
    st.write("En esta sección definiremos los parámetros para la Prueba Z.")
    
    col_h, col_sig = st.columns(2)
    
    with col_h:
        st.subheader("3.1. Definición de Hipótesis")
        mu_0 = st.number_input("Hipótesis Nula (H0: μ = )", value=50.0)
        tipo_test = st.selectbox("Hipótesis Alternativa (H1)", 
                                ["Bilateral (μ ≠ μ0)", "Cola Izquierda (μ < μ0)", "Cola Derecha (μ > μ0)"])
    
    with col_sig:
        st.subheader("3.2. Nivel de Significancia")
        alpha = st.select_slider("Selecciona el valor de α:", options=[0.01, 0.05, 0.10], value=0.05)
        st.write(f"Nivel de confianza: {(1-alpha)*100}%")
    
    # --- CONTINUACIÓN DEL MÓDULO 3: CÁLCULOS ---
    if st.button("Calcular Prueba de Hipótesis"):
        st.markdown("---")
        st.subheader("3.3. Estadístico de Prueba y Resultados")
        
        # 1. Obtener datos de la columna seleccionada
        datos_serie = df[col_analisis]
        n = len(datos_serie)
        media_muestral = datos_serie.mean()
        # Usamos la desviación estándar de la muestra como estimación de sigma
        sigma = datos_serie.std() 
        
        # 2. Cálculo del Estadístico Z
        # Fórmula: Z = (media_muestral - mu_0) / (sigma / sqrt(n))
        z_stat = (media_muestral - mu_0) / (sigma / np.sqrt(n))
        
        # 3. Cálculo del P-Value según el tipo de prueba
        if tipo_test == "Bilateral (μ ≠ μ0)":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            tipo_display = "Bilateral"
        elif tipo_test == "Cola Izquierda (μ < μ0)":
            p_value = stats.norm.cdf(z_stat)
            tipo_display = "Cola Izquierda"
        else: # Cola Derecha
            p_value = 1 - stats.norm.cdf(z_stat)
            tipo_display = "Cola Derecha"

        # Mostrar resultados (Sección 3.4 del reporte)
        res1, res2 = st.columns(2)
        with res1:
            st.metric("Media Muestral (x̄)", f"{media_muestral:.4f}")
            st.metric("Desviación Estándar (s)", f"{sigma:.4f}")
        with res2:
            st.metric("Tamaño de muestra (n)", n)
            st.metric("Estadístico Z", f"{z_stat:.4f}")

        st.write(f"**P-value calculado:** {p_value:.4f}")

        # --- 3.5. DECISIÓN ESTADÍSTICA ---
        st.subheader("3.5. Decisión")
        if p_value < alpha:
            st.error(f"RECHAZAR H0: El p-value ({p_value:.4f}) es menor que alpha ({alpha}).")
            st.write("Existen pruebas suficientes para decir que la media es distinta a la hipótesis planteada.")
        else:
            st.success(f"NO RECHAZAR H0: El p-value ({p_value:.4f}) es mayor o igual a alpha ({alpha}).")
            st.write("No hay pruebas suficientes para rechazar la hipótesis nula.")
            
        # Visualización de la zona de rechazo (Concepto clave)
        st.info("💡 Tip para tu reporte: La decisión se basa en comparar el P-value con el nivel de significancia seleccionado.")

        # GUARDAR EN MEMORIA (Añade estas líneas al final del bloque del botón)
        st.session_state['calculo_realizado'] = True
        st.session_state['datos_ia'] = {
            'col': col_analisis,
            'media': media_muestral,
            'mu0': mu_0,
            'n': n,
            'z': z_stat,
            'p': p_value,
            'alpha': alpha,
            'tipo': tipo_test
        }

   # --- 4. ASISTENTE DE IA (GEMINI API) ---
    st.markdown("---")
    st.header("4. Asistente de IA (Gemini API)")

    api_key = st.text_input("Introduce tu Gemini API Key:", type="password")

    if api_key and st.session_state.get('calculo_realizado'):
        if st.button("Pedir análisis a la IA"):
            try:
                # Asegúrate de usar la librería correcta y el modelo vigente
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
            
                # Datos calculados
                d = st.session_state['datos_ia']
            
                # PROMPT MEJORADO: Estructura profesional
                prompt = f"""
                Actúa como un experto en estadística inferencial. Analiza los resultados de la siguiente prueba de hipótesis Z:
            
                - Media muestral: {d['media']:.4f}
                - Estadístico Z: {d['z']:.4f}
                - P-value: {d['p']:.4f}
                - Nivel de significancia (alpha): {d['alpha']:.2f}
            
                Por favor, proporciona:
                1. Una interpretación clara de si se rechaza o no la hipótesis nula.
                2. Una explicación sencilla de qué significa el p-value obtenido en este contexto.
                """
            
                with st.spinner("Analizando con Gemini..."):
                    response = model.generate_content(prompt)
                    st.write("### Análisis estadístico:")
                    st.info(response.text)
                
            except Exception as e:
                st.error(f"Error técnico: {e}")
                st.write("Asegúrate de que tu API Key sea válida y empiece por 'AIza'.")

st.markdown("---")
st.caption("Proyecto de Probabilidad y Estadística - Entrega 18 de abril")