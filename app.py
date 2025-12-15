"""
This module implements a Streamlit web application for generating creative content.

The application allows users to generate blog articles, adapt them for Twitter,
and create accompanying cover images based on a given topic and target audience.

It utilizes language model chains for content generation and adaptation, and a
separate model for image generation.

To run the application:
streamlit run app.py
"""
import streamlit as st
from src.core.content_chains import create_blog_chain_groq, create_twitter_adaptor_chain, create_image_prompt_chain
from src.models.image_generator import generate_image_from_prompt

try:
    blog_chain = create_blog_chain_groq()
    image_prompt_chain = create_image_prompt_chain()
    twitter_adaptor_chain = create_twitter_adaptor_chain() 
except Exception as e:
    st.error(f"No se pudo inicializar un componente. Error: {e}")
    st.stop()


with st.sidebar:
    st.header("Parámetros de Generación")
    topic = st.text_input("Tema del Contenido:", "El impacto de la IA en la creatividad")
    audience = st.text_input("Audiencia Objetivo:", "Diseñadores gráficos y artistas digitales")
    
    generate_twitter = st.checkbox("Adaptar a Twitter/X", value=True) 
    generate_image = st.checkbox("Generar Imagen de Portada", value=False)
    
    generate_button = st.button("Generar Todo el Contenido", type="primary")

st.subheader("Contenido Generado")

if generate_button:
    if not topic or not audience:
        st.warning("Por favor, introduce el Tema y la Audiencia.")
    else:
        with st.spinner("Generando Artículo de Blog..."):
            inputs = {"topic": topic, "audience": audience}
            blog_content = blog_chain.invoke(inputs)
            
            st.markdown("### 📝 Artículo de Blog (Contenido Base)")
            st.markdown(blog_content)
        
        st.divider()

        if generate_twitter:
            st.markdown("### 🐦 Adaptación para Twitter/X")
            with st.spinner("Adaptando contenido a formato Twitter/X..."):
                twitter_inputs = {"blog_content": blog_content}
                twitter_content = twitter_adaptor_chain.invoke(twitter_inputs)
                st.markdown(twitter_content)
            st.divider()

        if generate_image:
            st.markdown("### 🖼️ Imagen de Portada")
            
            with st.spinner("1/2: Generando prompt de imagen..."):
                image_prompt = image_prompt_chain.invoke(inputs)
                st.info(f"**Prompt de Imagen Generado:** `{image_prompt}`")
            
            with st.spinner("2/2: Generando la imagen (Hugging Face API)..."):
                generated_image = generate_image_from_prompt(image_prompt)
                
                if generated_image:
                    st.image(generated_image, caption=topic, width='stretch')
                else:
                    st.warning("No se pudo generar la imagen.")