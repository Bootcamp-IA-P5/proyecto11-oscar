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
    create_blog_chain, create_image_prompt_chain,
    create_instagram_adaptor_chain, create_linkedin_adaptor_chain,
    create_twitter_adaptor_chain, generate_science_post_chain
)
from src.models.image_generator import (
    generate_image_from_huggingface,
    generate_image_from_replicate,
    search_image_from_unsplash,
    search_image_from_pexels
)
from src.core.rag_engine import ScienceRAG

st.set_page_config(layout="wide")

@st.cache_resource
def get_rag_engine():
    return ScienceRAG()

@st.cache_data
def get_cached_papers(_rag_engine):
    """
    _rag_engine empieza con guion bajo para que Streamlit 
    no intente hashear el objeto (ya que no cambia la salida).
    """
    return _rag_engine.list_indexed_papers()


def render_sidebar():
    """
    Renders the sidebar UI and collects user input for content generation.

    This function creates all the interactive widgets in the sidebar, allowing
    the user to specify brand information, choose a language model, set the
    generation language, define the topic and audience, and select which
    content pieces to generate (e.g., social media adaptations, cover image).

    Returns:
        tuple: A tuple containing all the user-selected settings:
            - brand_bio (str): Information about the company or person.
            - llm_selection (str): The chosen language model provider.
            - target_language (str): The language for the generated content.
            - topic (str): The main topic for the content.
            - audience (str): The target audience for the content.
            - generate_twitter (bool): Flag to generate a Twitter adaptation.
            - generate_instagram (bool): Flag to generate an Instagram adaptation.
            - generate_linkedin (bool): Flag to generate a LinkedIn adaptation.
            - generate_image (bool): Flag to generate a cover image.
            - image_provider (str or None): The selected image generation provider.
            - generate_button (bool): True if the generation button was clicked.
    """
    rag = get_rag_engine()
    
    st.sidebar.title("Generador de contenidos")
    
    with st.sidebar.expander("🔬 Base de Datos Científica"):
        topic_arxiv = st.text_input("Tema de investigación", value="LLM Safety")
        num_papers = st.slider("Cantidad de papers (Máx 3 para evitar errores)", 1, 3, 1)
        
        if st.button("Actualizar Conocimiento"):
            with st.spinner("Descargando y procesando..."):
                status = rag.ingest_papers(topic_arxiv, max_results=num_papers)
                st.cache_data.clear()
                st.success(status)
                st.rerun()
            
    with st.sidebar.expander("📄 Documentos disponibles"):
        indexed_papers = get_cached_papers(rag)
        
        if indexed_papers:
            st.write(f"Hay **{len(indexed_papers)}** documentos indexados:")
            for i, doc in enumerate(indexed_papers, start=1):
                st.markdown(f"{i}. {doc}")
                
            if st.button("Limpiar Base de Datos"):
                rag.reset_database()
                st.warning("Base de datos borrada.")
                st.rerun()
        else:
            st.warning("No hay documentos indexados.")
            
    with st.sidebar.expander("⚙️ Parametros de generación"):
        language_map = {
            "Español": "Spanish",
            "Inglés": "English",
            "Francés": "French",
            "Italiano": "Italian",
            "Japonés": "Japanese",
        }
        
        llm_options = [
            "Gemini (Nube, Prioritario)",
            "Groq (Nube, Rápido)",
            "Ollama (Local, Requiere setup)",
        ]
        
        st.header("💼 Información de la Empresa/Persona")
        brand_bio = st.text_area(
            "Datos de la empresa/persona:",
            placeholder="Ej: Somos una agencia de marketing especializada en IA...",
            help="Esta información se usará para personalizar el tono y el contenido.",
        )
        
        st.header("🧠 Motor de generación (LLM)")
        llm_selection_display = st.selectbox(
            "Seleccionar Proveedor",
            options=llm_options,
            index=1,
            help="Gemini: Versátil; Groq: Ultra-rápido; Ollama: Requiere servidor local.",
        )
        llm_selection = llm_selection_display.split("(")[0].strip()

        if llm_selection == "Ollama":
            st.warning(
                "⚠️ **ADVERTENCIA:** Ollama requiere un servidor local en el puerto "
                "11434 con el modelo necesario descargado."
            )
            
        st.header("🌐 Idiomas")
        selected_language = st.selectbox(
            "Idioma a usar para la generación",
            options=list(language_map.keys()),
            index=0,
            help="Idioma en el que se generará el contenido base y sus adaptaciones.",
        )
        target_language = language_map[selected_language]
            
    with st.sidebar:
        topic = st.text_input(
            "Tema del Contenido:", "Agujeros negros"
        )
        audience = st.text_input(
            "Audiencia Objetivo:", "Todo el público"
        )

        generate_twitter = st.checkbox("Adaptar a Twitter/X", value=False)
        generate_instagram = st.checkbox("Adaptar a Instagram", value=False)
        generate_linkedin = st.checkbox("Adaptar a LinkedIn", value=False)
        generate_image = st.checkbox("Generar imagen de portada", value=False)

        image_provider = None
        if generate_image:
            image_provider = st.selectbox(
                "Proveedor de Imagen",
                [
                    "Unsplash (Stock Photos)",
                    "Pexels (Stock Photos)",
                    "Hugging Face (SDXL)",
                    "Replicate (Flux)"
                ],
                help="Unsplash y Pexels buscan fotos reales de alta calidad. HuggingFace y Replicate generan imágenes con IA."
            )
        else:
            st.info("💡 Solo se generará el contenido de texto.")

        generate_button = st.button("Generar Todo el Contenido", type="primary")

    return (
        brand_bio,
        llm_selection,
        target_language,
        topic,
        audience,
        generate_twitter,
        generate_instagram,
        generate_linkedin,
        generate_image,
        image_provider,
        generate_button,
    )


def generate_content(
    brand_bio,
    llm_selection,
    target_language,
    topic,
    audience,
    generate_twitter,
    generate_instagram,
    generate_linkedin,
    generate_image,
    image_provider,
):
    """
    Generates and displays content based on the user's selections.

    This function orchestrates the content generation process. It initializes the
    required language model chains, generates a base blog article, and then creates
    adaptations for social media platforms and a cover image if requested.

    Args:
        brand_bio (str): The brand biography provided by the user.
        llm_selection (str): The selected language model (e.g., "Gemini", "Groq").
        target_language (str): The target language for the content.
        topic (str): The main topic for the content generation.
        audience (str): The target audience for the content.
        generate_twitter (bool): If True, generates content for Twitter/X.
        generate_instagram (bool): If True, generates content for Instagram.
        generate_linkedin (bool): If True, generates content for LinkedIn.
        generate_image (bool): If True, generates a cover image.
        image_provider (str or None): The provider for the image generation service.
    """
    if not topic or not audience:
        st.warning("Por favor, introduce el Tema y la Audiencia.")
        return

    try:
        st.info(f"Inicializando motor de contenido con: **{llm_selection}**...")
        blog_chain = create_blog_chain(llm_selection)
        image_prompt_chain = create_image_prompt_chain(llm_selection)
        twitter_adaptor_chain = create_twitter_adaptor_chain(llm_selection)
        instagram_adaptor_chain = create_instagram_adaptor_chain(llm_selection)
        linkedin_adaptor_chain = create_linkedin_adaptor_chain(llm_selection)
        science_post_chain = generate_science_post_chain(llm_selection)
        
    except Exception as e:
        st.error(f"No se pudo inicializar un componente. Error: {e}")
        st.stop()

    brand_bio = brand_bio.strip() if brand_bio.strip() else "No proporcionado."

    with st.spinner("Generando Artículo de Blog..."):
        st.info(f"Recuperando información de la base de datos científica...")
        rag = ScienceRAG()
        documents = rag.get_context(topic)
        
        if not documents:
            st.warning("No se han encontrado documentos relevantes.")
            
            inputs = {
                "topic": topic,
                "audience": audience,
                "target_language": target_language,
                "brand_bio": brand_bio,
            }
            blog_content = blog_chain.invoke(inputs)
            st.markdown("### 📝 Artículo de Blog")
            st.markdown(blog_content)
        else:
            blog_content = science_post_chain.invoke(
                {
                    "documents": documents, 
                    "topic": topic, 
                    "target_language": target_language,
                    "brand_bio": brand_bio                
                }
            )
            st.markdown("### 📝 Artículo de Blog Científico")
            st.markdown(blog_content)

    st.divider()

    if generate_twitter:
        st.markdown("### 🐦 Adaptación para Twitter/X")
        with st.spinner("Adaptando contenido a formato Twitter/X..."):
            twitter_inputs = {
                "blog_content": blog_content,
                "target_language": target_language,
                "brand_bio": brand_bio,
            }
            twitter_content = twitter_adaptor_chain.invoke(twitter_inputs)
            st.markdown(twitter_content)
        st.divider()

    if generate_instagram:
        st.markdown("### 📸 Adaptación para Instagram")
        with st.spinner("Adaptando contenido a Instagram (Caption)..."):
            insta_inputs = {
                "blog_content": blog_content,
                "target_language": target_language,
                "brand_bio": brand_bio,
            }
            instagram_content = instagram_adaptor_chain.invoke(insta_inputs)
            st.markdown(instagram_content)
        st.divider()

    if generate_linkedin:
        st.markdown("### 💼 Adaptación para LinkedIn")
        with st.spinner("Adaptando contenido a LinkedIn..."):
            linkedin_inputs = {
                "blog_content": blog_content,
                "target_language": target_language,
                "brand_bio": brand_bio,
            }
            linkedin_content = linkedin_adaptor_chain.invoke(linkedin_inputs)
            st.markdown(linkedin_content)
        st.divider()

    if generate_image and image_provider:
        with st.spinner(f"Renderizando imagen con {image_provider}..."):
            # Determine which provider to use
            image_result = None
            
            # For stock photo APIs, use the topic directly instead of generating an AI prompt
            if "Unsplash" in image_provider or "Pexels" in image_provider:
                # Use the topic for keyword-based search
                search_query = topic
                if "Unsplash" in image_provider:
                    image_result = search_image_from_unsplash(search_query)
                else:  # Pexels
                    image_result = search_image_from_pexels(search_query)
            else:
                # For AI image generation, use the detailed prompt from LLM
                img_prompt = image_prompt_chain.invoke({"blog_content": blog_content})
                if "Replicate" in image_provider:
                    path = generate_image_from_replicate(img_prompt)
                    if path:
                        image_result = path
                else:  # HuggingFace
                    image_result = generate_image_from_huggingface(img_prompt)

            if image_result:
                st.image(image_result, caption=f"Imagen obtenida vía {image_provider}", use_container_width=True)
            else:
                st.error(f"No se pudo obtener la imagen desde {image_provider}")


def main():
    """
    Main function to run the Streamlit application.

    This function sets up the main UI, renders the sidebar to get user inputs,
    and triggers the content generation process when the user clicks the button.
    """
    st.subheader("Contenido Generado (PoC)")

    (
        brand_bio,
        llm_selection,
        target_language,
        topic,
        audience,
        generate_twitter,
        generate_instagram,
        generate_linkedin,
        generate_image,
        image_provider,
        generate_button,
    ) = render_sidebar()

    if generate_button:
        generate_content(
            brand_bio,
            llm_selection,
            target_language,
            topic,
            audience,
            generate_twitter,
            generate_instagram,
            generate_linkedin,
            generate_image,
            image_provider,
        )


if __name__ == "__main__":
    main()
