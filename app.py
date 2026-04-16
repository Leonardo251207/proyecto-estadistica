import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from scipy import stats

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

st.markdown("---")
st.caption("Proyecto de Probabilidad y Estadística - Entrega 18 de abril")