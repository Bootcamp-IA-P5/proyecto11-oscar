import streamlit as st
from src.core.content_chains import create_blog_chain_ollama

st.set_page_config(page_title="Generador de Contenido PoC", layout="wide")
st.title("✍️ Digital Content Generator (PoC)")
st.caption("Generación de contenido de blog con modelos LLM locales (Ollama).")

try:
    blog_chain = create_blog_chain_ollama(model_name="mistral") # Asegúrate de que el modelo esté descargado en Ollama
except Exception as e:
    st.error(f"No se pudo inicializar el LLM. Asegúrate de que Ollama esté corriendo y el modelo (ej: llama3:8b) esté descargado. Error: {e}")
    st.stop()


with st.sidebar:
    st.header("Parámetros de Generación")
    topic = st.text_input("Tema del Blog:", "El impacto de la IA generativa en el marketing digital")
    audience = st.text_input("Audiencia Objetivo:", "Dueños de PYMES y especialistas en marketing")
    
    generate_button = st.button("Generar Contenido de Blog", type="primary")

st.subheader("Contenido Generado (Plataforma: Blog)")

if generate_button:
    if not topic or not audience:
        st.warning("Por favor, introduce el Tema y la Audiencia.")
    else:
        with st.spinner("Generando contenido... esto puede tardar un momento con modelos locales."):
            try:
                inputs = {"topic": topic, "audience": audience}
                result = blog_chain.invoke(inputs)
                
                st.markdown(result)
                
            except Exception as e:
                st.error(f"Ocurrió un error durante la generación: {e}")