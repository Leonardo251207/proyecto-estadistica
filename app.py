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

    st.markdown("---")
    # --- 3. PRUEBA DE HIPÓTESIS (Z-TEST) ---
    st.header("3. Prueba de Hipótesis para la Media (Z-Test)")
    
    col1, col2 = st.columns(2)
    with col1:
        mu_0 = st.number_input("Valor hipotético de la media (μ₀):", value=float(df[col_analisis].mean()))
    with col2:
        alpha = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05)
        
    tipo_test = st.selectbox("Tipo de prueba:", ["Bilateral (μ ≠ μ0)", "Cola Izquierda (μ < μ0)", "Cola Derecha (μ > μ0)"])

    # --- BOTÓN DE CÁLCULO ---
    if st.button("Calcular Prueba de Hipótesis"):
        datos_serie = df[col_analisis]
        n = len(datos_serie)
        media_muestral = datos_serie.mean()
        sigma = datos_serie.std() 
        
        # Cálculo de Z
        z_stat = (media_muestral - mu_0) / (sigma / np.sqrt(n))
        
        # Cálculo de P-Value
        if tipo_test == "Bilateral (μ ≠ μ0)":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif tipo_test == "Cola Izquierda (μ < μ0)":
            p_value = stats.norm.cdf(z_stat)
        else:
            p_value = 1 - stats.norm.cdf(z_stat)

        # GUARDAR EN SESSION STATE (Para que no se borre al usar la IA)
        st.session_state['datos_ia'] = {
            'col_nombre': col_analisis,
            'media': media_muestral,
            'sigma': sigma,
            'n': n,
            'z': z_stat,
            'p': p_value,
            'alpha': alpha,
            'mu0': mu_0
        }
        st.session_state['calculo_realizado'] = True

    # --- MOSTRAR RESULTADOS (FUERA DEL BOTÓN PARA PERSISTENCIA) ---
    if st.session_state.get('calculo_realizado', False):
        d = st.session_state['datos_ia']
        
        st.markdown("---")
        st.subheader("3.3. Estadístico de Prueba y Resultados")
        
        res1, res2 = st.columns(2)
        with res1:
            st.metric("Media Muestral (x̄)", f"{d['media']:.4f}")
            st.metric("Desviación Estándar (s)", f"{d['sigma']:.4f}")
        with res2:
            st.metric("Tamaño de muestra (n)", d['n'])
            st.metric("Estadístico Z", f"{d['z']:.4f}")

        st.write(f"**P-value calculado:** {d['p']:.4f}")

        # --- 3.5. DECISIÓN ESTADÍSTICA ---
        st.subheader("3.5. Decisión")
        if d['p'] < d['alpha']:
            st.error(f"RECHAZAR H0: El p-value ({d['p']:.4f}) es menor que alpha ({d['alpha']}).")
            st.write("**Conclusión:** Existen pruebas suficientes para decir que la media es distinta a la hipótesis planteada.")
        else:
            st.success(f"NO RECHAZAR H0: El p-value ({d['p']:.4f}) es mayor o igual a alpha ({d['alpha']}).")
            st.write("**Conclusión:** No hay pruebas suficientes para rechazar la hipótesis nula.")

        st.markdown("---")
        
        # --- 4. EXPLICACIÓN CON IA ---
        st.header("4. Análisis con Inteligencia Artificial")
        
        # Mostramos los datos que la IA va a leer para que el usuario esté seguro
        st.info(f"Análisis para la variable: **{d['col_nombre']}**")
        
        api_key = st.text_input("Introduce tu Google API Key (Gemini):", type="password")
        
        if st.button("Generar Razonamiento IA"):
            if not api_key:
                st.warning("⚠️ Por favor, introduce una API Key válida.")
            else:
                try:
                    genai.configure(api_key=api_key)
                    # IMPORTANTE: gemini-1.5-flash es el modelo vigente y estable
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    # PROMPT ESTILO ACADÉMICO (Según guías docentes)
                    prompt = f"""
                    Actúa como un experto en estadística inferencial. Analiza los siguientes resultados:
                    - Variable: {d['col_nombre']}
                    - Media obtenida: {d['media']:.4f} vs Media Hipotética (mu0): {d['mu0']:.4f}
                    - Estadístico Z: {d['z']:.4f}
                    - P-value: {d['p']:.4f}
                    - Alpha: {d['alpha']:.2f}

                    Por favor, proporciona una respuesta estructurada en:
                    1. Interpretación del Estadístico Z (distancia en desviaciones estándar).
                    2. Explicación del P-value y su relación con el nivel de significancia.
                    3. Conclusión final en lenguaje sencillo pero profesional.
                    """
                    
                    with st.spinner("Gemini está procesando el razonamiento..."):
                        response = model.generate_content(prompt)
                        st.write("### 🤖 Razonamiento del Experto IA:")
                        st.info(response.text)
                        
                except Exception as e:
                    st.error(f"Error técnico: {e}")
                    st.write("Verifica tu API Key y conexión.")

st.markdown("---")
st.caption("Proyecto de Probabilidad y Estadística - Entrega Final")