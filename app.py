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
    create_twitter_adaptor_chain, generate_science_post_chain,
    assemble_grounding_context, get_grounding_summary
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
            - finance_enabled (bool): Flag to enable financial news.
            - finance_query (str or None): Query for financial news.
            - finance_max_articles (int): Maximum number of financial articles.
            - generate_twitter (bool): Flag to generate a Twitter adaptation.
            - generate_instagram (bool): Flag to generate an Instagram adaptation.
            - generate_linkedin (bool): Flag to generate a LinkedIn adaptation.
            - generate_image (bool): Flag to generate a cover image.
            - image_provider (str or None): The selected image generation provider.
            - generate_button (bool): True if the generation button was clicked.
    """
    rag = get_rag_engine()
    
    st.sidebar.title("Generador de contenidos")
    
    # Sección de Noticias Financieras (opcional)
    with st.sidebar.expander("📈 Noticias Financieras (opcional)"):
        finance_enabled = st.checkbox(
            "Activar noticias financieras",
            value=False,
            help="Incluye noticias financieras en tiempo real desde Alpha Vantage"
        )
        
        finance_query = None
        finance_max_articles = 5
        
        if finance_enabled:
            finance_query = st.text_input(
                "Tema financiero o empresa",
                placeholder="Ej: Tesla, inflación, tipos de interés",
                help="Consulta para buscar noticias financieras"
            )
            finance_max_articles = st.slider(
                "Máx. artículos",
                min_value=1,
                max_value=10,
                value=5,
                help="Número máximo de noticias a recuperar"
            )
            # Opción de depuración: mostrar respuesta cruda de la API
            show_finance_debug = st.checkbox(
                "Mostrar respuesta cruda de la API financiera (debug)",
                value=False,
                help="Muestra la respuesta de Alpha Vantage y cómo se detecta ticker vs tema. Solo para depuración."
            )
            if show_finance_debug and finance_query:
                try:
                    from src.models.financial_news import load_financial_news
                    st.info("Detectando tipo de consulta: ticker (mayúsculas/formato) vs tema (texto libre)")
                    import re
                    is_ticker = bool(re.match(r'^[A-Z0-9:_\-]+$', finance_query.strip().upper()))
                    st.write("Detectado como:", "Ticker" if is_ticker else "Tema/texto")
                    raw_news = load_financial_news(finance_query, max_articles=finance_max_articles)
                    if raw_news:
                        st.write("Artículos devueltos:", len(raw_news))
                        st.json(raw_news)
                    else:
                        st.warning("No se devolvieron artículos. Comprueba la API key, los límites de la API o prueba otra consulta.")
                except Exception as _err:
                    st.error(f"Error al recuperar noticias financieras: {_err}")

    # Sección de Base Científica (opcional)
    with st.sidebar.expander("🔬 Base de Datos Científica (arXiv)"):
        rag_enabled = st.checkbox(
            "Activar contexto científico (arXiv)",
            value=True,
            help="Incluye contexto científico recuperado de papers en arXiv"
        )
        topic_arxiv = st.text_input("Tema de investigación", value="LLM Safety")
        num_papers = st.slider("Cantidad de papers (Máx 3 para evitar errores)", 1, 3, 1)
        
        if st.button("Actualizar Conocimiento"):
            with st.spinner("Descargando y procesando..."):
                rag = get_rag_engine()
                status = rag.ingest_papers(topic_arxiv, max_results=num_papers)
                st.cache_data.clear()
                st.success(status)
                st.rerun()
    
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
                st.cache_data.clear()
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
        
        # Debug mode for developers
        with st.expander("🔧 Developer Options"):
            debug_mode = st.checkbox(
                "Enable debug mode",
                value=False,
                help="Show grounding context details in logs and UI"
            )
        
        generate_button = st.button("Generar Todo el Contenido", type="primary")

    return (
        brand_bio,
        llm_selection,
        target_language,
        topic,
        audience,
        finance_enabled,
        finance_query,
        finance_max_articles,
        generate_twitter,
        generate_instagram,
        generate_linkedin,
        generate_image,
        image_provider,
        debug_mode,
        generate_button,
    )


def generate_content(
    brand_bio,
    llm_selection,
    target_language,
    topic,
    audience,
    finance_enabled,
    finance_query,
    finance_max_articles,
    generate_twitter,
    generate_instagram,
    generate_linkedin,
    generate_image,
    image_provider,
    debug_mode=False,
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
        finance_enabled (bool): If True, includes financial news context.
        finance_query (str or None): Query for financial news API.
        finance_max_articles (int): Maximum number of financial articles to fetch.
        generate_twitter (bool): If True, generates content for Twitter/X.
        generate_instagram (bool): If True, generates content for Instagram.
        generate_linkedin (bool): If True, generates content for LinkedIn.
        generate_image (bool): If True, generates a cover image.
        image_provider (str or None): The provider for the image generation service.
        debug_mode (bool): If True, shows debug information about grounding context.
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
    
    # Prepare financial context if enabled (now returns tuple)
    from src.core.content_chains import _prepare_financial_context
    financial_context, financial_articles = _prepare_financial_context(
        use_finance=finance_enabled,
        finance_query=finance_query,
        finance_max_articles=finance_max_articles
    )

    # Recuperar contexto científico solo si está activado
    documents = ""
    if 'rag_enabled' in locals() and rag_enabled:
        rag = ScienceRAG()
        documents = rag.get_context(topic)

    with st.spinner("Generando Artículo de Blog..."):
        # Resumen de grounding para transparencia en la UI
        grounding_summary = get_grounding_summary(
            rag_documents=documents,
            financial_articles=financial_articles
        )
        # Ensamblar contexto combinado con logging si debug
        combined_context = assemble_grounding_context(
            rag_context=documents,
            financial_context=financial_context,
            debug=debug_mode
        )

        # TRANSPARENCIA UI: Mostrar fuentes utilizadas
        if grounding_summary["is_grounded"]:
            st.success(f"📡 **Contenido fundamentado en fuentes externas:** {', '.join(grounding_summary['sources_used'])}")
            # Mostrar artículos financieros recuperados
            if grounding_summary["financial_enabled"]:
                st.markdown(f"#### 📈 Noticias Financieras ({grounding_summary['financial_article_count']} artículos)")
                for article in grounding_summary["financial_articles"]:
                    st.markdown(f"- **{article['title']}**  
                        <span style='color:gray;font-size:small'>Fuente: {article['source']}</span>", unsafe_allow_html=True)
                if not grounding_summary["financial_articles"]:
                    st.warning("No se recuperaron noticias financieras.")
            # Mostrar papers científicos recuperados
            if grounding_summary["rag_enabled"]:
                st.markdown(f"#### 🔬 Contexto Científico (arXiv): {grounding_summary['rag_doc_count']} fragmentos de papers")
                if not documents:
                    st.warning("No se recuperó contexto científico de arXiv.")
            # Debug: mostrar contexto ensamblado
            if debug_mode:
                st.markdown("---")
                st.markdown("**🔧 Debug: Contexto ensamblado (primeros 1000 chars)**")
                st.code(combined_context[:1000] if combined_context else "[Contexto vacío]", language="text")
        else:
            st.info("🧠 Generando con LLM puro (sin fuentes externas)")

        # Selección de cadena según disponibilidad de contexto científico
        if not documents:
            inputs = {
                "topic": topic,
                "audience": audience,
                "target_language": target_language,
                "brand_bio": brand_bio,
                "financial_context": financial_context
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
                    "brand_bio": brand_bio,
                    "financial_context": financial_context
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
        finance_enabled,
        finance_query,
        finance_max_articles,
        generate_twitter,
        generate_instagram,
        generate_linkedin,
        generate_image,
        image_provider,
        debug_mode,
        generate_button,
    ) = render_sidebar()

    if generate_button:
        generate_content(
            brand_bio,
            llm_selection,
            target_language,
            topic,
            audience,
            finance_enabled,
            finance_query,
            finance_max_articles,
            generate_twitter,
            generate_instagram,
            generate_linkedin,
            generate_image,
            image_provider,
            debug_mode,
        )



if __name__ == "__main__":
    main()
