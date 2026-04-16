import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Configuración de la página (opcional, pero se ve más pro)
st.set_page_config(page_title="Análisis Estadístico", layout="wide")

# Título de la App (Sección 1 del reporte)
st.title("📊 Aplicación de Análisis Estadístico y Pruebas de Hipótesis")
st.markdown("---")

# --- 1. MÓDULO DE CARGA DE DATOS ---
st.header("1. Carga de Datos")
tipo_datos = st.radio("Selecciona origen de datos:", ["Generación sintética", "Cargar CSV"])

df = None

if tipo_datos == "Generación sintética":
    # Generamos datos con una distribución normal (loc=media, scale=desviación)
    datos = np.random.normal(loc=50, scale=10, size=100)
    df = pd.DataFrame(datos, columns=["Variable_X"])
    st.success("✅ Datos sintéticos generados correctamente (n=100)")
else:
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.success("✅ Archivo cargado con éxito")

# Solo ejecutamos el resto si hay datos cargados
if df is not None:
    st.markdown("---")
    st.write("### Vista previa de los datos")
    st.dataframe(df.head(), use_container_width=True)

    # --- 2. DESCRIPCIÓN DE LOS DATOS (HISTOGRAMA PROFESIONAL) ---
    st.header("2. Descripción de los Datos")
    
    col_analisis = st.selectbox("Selecciona la variable para analizar:", df.columns)
    
    # Creamos columnas para organizar la gráfica y las preguntas
    col_grafica, col_preguntas = st.columns([2, 1])
    
    with col_grafica:
        # Configuración de estilo profesional
        plt.style.use('ggplot') 
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histograma mejorado
        sns.histplot(
            df[col_analisis], 
            kde=True, 
            ax=ax, 
            color="#1f77b4",  # Azul profesional
            edgecolor="white", 
            linewidth=1.5,
            alpha=0.7
        )
        
        # Personalización de títulos y etiquetas
        ax.set_title(f"Distribución de Frecuencias: {col_analisis}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(col_analisis, fontsize=11)
        ax.set_ylabel("Frecuencia", fontsize=11)
        
        # Mostrar gráfica
        st.pyplot(fig)

        # --- MÓDULO DE DESCARGA ---
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button(
            label="💾 Descargar Histograma en PNG",
            data=buf.getvalue(),
            file_name=f"histograma_{col_analisis}.png",
            mime="image/png"
        )

    with col_preguntas:
        st.write("### 📝 Análisis Visual")
        st.info("Responde estas preguntas según lo observado en la gráfica para tu reporte.")
        
        norm_obs = st.radio("¿La distribución parece normal?", ["Sí", "No"], key="obs_norm")
        sesgo_obs = st.radio("¿Se observa algún sesgo?", ["No", "Izquierda", "Derecha"], key="obs_sesgo")
        outliers_obs = st.radio("¿Se observan valores atípicos (outliers)?", ["No", "Sí"], key="obs_out")

    st.markdown("---")
    st.caption("Proyecto de Probabilidad y Estadística - Desarrollado para entrega el 18 de abril.")