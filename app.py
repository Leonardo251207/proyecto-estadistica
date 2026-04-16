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

    # --- 2. DESCRIPCIÓN DE LOS DATOS (FILTRO DE VARIABLES CUANTITATIVAS) ---
    st.header("2. Descripción de los Datos")
    
    # FILTRO CRÍTICO: Solo permitimos columnas numéricas (int y float)
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columnas_numericas) > 0:
        col_analisis = st.selectbox("Selecciona la variable numérica para analizar:", columnas_numericas)
        
        col_grafica, col_preguntas = st.columns([2, 1])
        
        with col_grafica:
            plt.style.use('ggplot') 
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Histograma profesional con KDE
            sns.histplot(
                df[col_analisis], 
                kde=True, 
                ax=ax, 
                color="#1f77b4", 
                edgecolor="white", 
                linewidth=1.5,
                alpha=0.7
            )
            
            ax.set_title(f"Distribución de: {col_analisis}", fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(col_analisis, fontsize=11)
            ax.set_ylabel("Frecuencia / Densidad", fontsize=11)
            
            st.pyplot(fig)

            # Botón de descarga
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            st.download_button(
                label="💾 Descargar Histograma Profesional",
                data=buf.getvalue(),
                file_name=f"histograma_{col_analisis}.png",
                mime="image/png"
            )

        with col_preguntas:
            st.write("### 📝 Análisis Visual")
            st.info("Utiliza esta sección para completar el punto 2 de tu reporte.")
            st.radio("¿La distribución parece normal?", ["Sí", "No"], key="obs_norm")
            st.radio("¿Se observa algún sesgo?", ["No", "Izquierda", "Derecha"], key="obs_sesgo")
            st.radio("¿Hay presencia de Outliers?", ["No", "Sí"], key="obs_out")
            
            # Datos para la Sección 2 del reporte
            st.write(f"**Estadísticas rápidas de {col_analisis}:**")
            st.write(f"- N. de observaciones: {len(df[col_analisis])}")
            st.write(f"- Media: {df[col_analisis].mean():.2f}")
    else:
        st.error("❌ El archivo cargado no contiene columnas numéricas. Por favor, sube un archivo con datos cuantitativos.")

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