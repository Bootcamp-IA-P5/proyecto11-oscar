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
from src.core.content_chains import (
    create_twitter_adaptor_chain, create_image_prompt_chain,
    create_instagram_adaptor_chain, create_linkedin_adaptor_chain, create_blog_chain
)
from src.models.image_generator import generate_image_from_prompt

with st.sidebar:
    st.header("Parámetros de Generación")
    
    st.subheader("🧠 Motor de Generación (LLM)")
    
    llm_options = ["Gemini (Nube, Prioritario)", "Groq (Nube, Rápido)", "Ollama (Local, Requiere setup)"]
    
    llm_selection_display = st.selectbox(
        "Seleccionar Proveedor",
        options=llm_options,
        index=1,
        help="Gemini: Versátil; Groq: Ultra-rápido; Ollama: Requiere servidor local Ollama."
    )
    
    llm_selection = llm_selection_display.split('(')[0].strip()
    
    # Advertencia específica si selecciona Ollama
    if llm_selection == "Ollama":
        st.warning(
            "⚠️ **ADVERTENCIA:** Ollama está seleccionado. Esta opción requiere que el servidor "
            "de Ollama esté corriendo localmente en el puerto 11434 y que el modelo necesario "
            "esté descargado. Si no está configurado, la aplicación fallará al generar."
        )
    
    # llm_selection = st.selectbox(
    #     "🧠 Modelo de Lenguaje (LLM)",
    #     options=["Gemini", "Groq"],
    #     index=1,
    #     help="Gemini: Versátil, Groq: Ultra-rápido."
    # )
    st.divider()
    
    topic = st.text_input("Tema del Contenido:", "El impacto de la IA en la creatividad")
    audience = st.text_input("Audiencia Objetivo:", "Diseñadores gráficos y artistas digitales")
    
    generate_twitter = st.checkbox("Adaptar a Twitter/X", value=False)
    generate_instagram = st.checkbox("Adaptar a Instagram", value=False)
    generate_linkedin = st.checkbox("Adaptar a LinkedIn", value=False)
    generate_image = st.checkbox("Generar Imagen de Portada", value=False)
    
    generate_button = st.button("Generar Todo el Contenido", type="primary")

st.subheader("Contenido Generado (PoC)")

if generate_button:
    if not topic or not audience:
        st.warning("Por favor, introduce el Tema y la Audiencia.")
    else:
        try:
            st.info(f"Inicializando motor de contenido con: **{llm_selection}**...")
            blog_chain = create_blog_chain(llm_selection)
            image_prompt_chain = create_image_prompt_chain(llm_selection)
            twitter_adaptor_chain = create_twitter_adaptor_chain(llm_selection)
            instagram_adaptor_chain = create_instagram_adaptor_chain(llm_selection)
            linkedin_adaptor_chain = create_linkedin_adaptor_chain(llm_selection)
        except Exception as e:
            st.error(f"No se pudo inicializar un componente. Error: {e}")
            st.stop()
        
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
            
        if generate_instagram:
            st.markdown("### 📸 3. Adaptación para Instagram")
            with st.spinner("Adaptando contenido a Instagram (Caption)..."):
                instagram_inputs = {"blog_content": blog_content}
                instagram_content = instagram_adaptor_chain.invoke(instagram_inputs)
                st.markdown(instagram_content)
            st.divider()

        if generate_linkedin:
            st.markdown("### 💼 4. Adaptación para LinkedIn")
            with st.spinner("Adaptando contenido a LinkedIn..."):
                linkedin_inputs = {"blog_content": blog_content}
                linkedin_content = linkedin_adaptor_chain.invoke(linkedin_inputs)
                st.markdown(linkedin_content)
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