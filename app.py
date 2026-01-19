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

st.set_page_config(
    page_title="Generador de Contenido IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal compacto
st.markdown("""
<div style="text-align: center; margin-bottom: 0.3rem;">
    <h1 style="color: #0F172A; margin: 0; font-size: 2rem; font-weight: 800;">🚀 Generador de Contenido IA</h1>
    <p style="color: #1E3A5F; margin: 0.2rem 0 0 0; font-size: 0.95rem;">Proyecto 11 - PXI G3</p>
</div>
""", unsafe_allow_html=True)

# Subtítulo de sección - alineado a la izquierda y compacto
st.markdown("""
<div style="margin: 0.2rem 0 0.5rem 0;">
    <p style="color: #334155; margin: 0; font-size: 1.7rem; font-weight: 700; text-align: left;">Contenido Generado (PoC)</p>
</div>
""", unsafe_allow_html=True)

# Custom CSS styling - Tech Light Theme (Presentación)
st.markdown("""
<style>
    /* Spinner morado visible con alta especificidad */
    .stSpinner > div,
    .stSpinner > div > div,
    div[data-testid="stSpinner"] > div,
    [class*="stSpinner"] > div {
        border-top-color: #9F7AEA !important;
        border-right-color: rgba(159, 122, 234, 0.3) !important;
        border-bottom-color: rgba(159, 122, 234, 0.3) !important;
        border-left-color: rgba(159, 122, 234, 0.3) !important;
    }
    
    /* Menú de configuración en negro */
    [data-testid="stToolbar"] {
        color: #000000 !important;
    }
    
    [data-testid="stToolbar"] button {
        color: #000000 !important;
    }
    
    /* Ocultar menús de Streamlit innecesarios */
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    
    /* Main background - Light blue gradient like presentation */
    .stApp {
        background: linear-gradient(135deg, #C8E0F4 0%, #A8C8E8 50%, #B8D4F0 100%) !important;
    }
    
    /* Sidebar styling - Azul marino oscuro */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2C3E5A 0%, #1E2A3A 50%, #16202E 100%) !important;
        border-right: 2px solid rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Sidebar text - Claro para contraste con fondo oscuro */
    [data-testid="stSidebar"] * {
        color: #E8F1F5 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown *,
    [data-testid="stSidebar"] label *,
    [data-testid="stSidebar"] .element-container *,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stExpander label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] div[class*="stMarkdown"] {
        color: #E8F1F5 !important;
    }
    
    /* Sidebar input fields and dropdowns - Fondo oscuro con morado */
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background: linear-gradient(135deg, #3A4A5E 0%, #2C3A4E 100%) !important;
        color: #E8F1F5 !important;
        border: 1px solid #9F7AEA !important;
    }
    
    /* Text input fields - white background con texto negro */
    [data-testid="stSidebar"] input[type="text"],
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] input:not([type="checkbox"]):not([type="radio"]) {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 500 !important;
        border: 2px solid #9F7AEA !important;
    }
    
    /* Placeholder text más marcado */
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #64748B !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }
    
    /* También para el área principal */
    input::placeholder,
    textarea::placeholder {
        color: #475569 !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }
    
    /* Selected value in dropdown */
    [data-testid="stSidebar"] [data-baseweb="select"] > div > div,
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {
        background: linear-gradient(135deg, #3A4A5E 0%, #2C3A4E 100%) !important;
        color: #E8F1F5 !important;
    }
    
    /* Spinner styling */
    [data-testid="stSidebar"] .stSpinner > div,
    [data-testid="stSidebar"] .stStatus,
    [data-testid="stSidebar"] [data-testid="stStatusWidget"],
    [data-testid="stSidebar"] .stAlert,
    [data-testid="stSidebar"] [data-testid="stNotification"],
    [data-testid="stSidebar"] .element-container {
        background-color: transparent !important;
        border: none !important;
        padding: 0.5rem;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] .stSpinner,
    [data-testid="stSidebar"] .stSpinner *,
    [data-testid="stSidebar"] .stStatus *,
    [data-testid="stSidebar"] [data-testid="stStatusWidget"] * {
        color: #E8F1F5 !important;
        background-color: transparent !important;
    }
    
    /* Hide Streamlit cache messages */
    .stCaching,
    [data-testid="stCaching"],
    [data-testid="stStreamlitDialog"],
    div[data-testid="stNotification"]:has(div:contains("Running")),
    .st-emotion-cache-ue6h4q,
    [data-testid="stStatusWidget"],
    .stStatus,
    div[data-testid="stStatus"],
    div[class*="stStatus"],
    div[class*="StatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        position: absolute !important;
        left: -9999px !important;
    }
    
    .stSpinner > div,
    .stSpinner {
        background: transparent !important;
    }
    
    /* Spinner morado en el contenido principal */
    .stSpinner > div,
    .stSpinner > div > div {
        border-top-color: #9F7AEA !important;
        border-right-color: rgba(159, 122, 234, 0.3) !important;
        border-bottom-color: rgba(159, 122, 234, 0.3) !important;
        border-left-color: rgba(159, 122, 234, 0.3) !important;
    }
    
    /* Help icons - cyan claro */
    [data-testid="stSidebar"] .stTooltipIcon,
    [data-testid="stSidebar"] button[kind="helpTooltip"],
    [data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] {
        color: #00D4FF !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Expander styling - oscuro con borde cyan */
    [data-testid="stSidebar"] .streamlit-expanderHeader,
    [data-testid="stSidebar"] [data-testid="stExpander"] > div,
    [data-testid="stSidebar"] details > summary,
    [data-testid="stSidebar"] .stExpander {
        background: linear-gradient(135deg, #3A4A5E 0%, #2C3A4E 100%) !important;
        color: #E8F1F5 !important;
        border: 1px solid #00D4FF80 !important;
        border-radius: 8px;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover,
    [data-testid="stSidebar"] details > summary:hover {
        background: linear-gradient(135deg, #4A5A6E 0%, #3C4A5E 100%) !important;
        color: #00D4FF !important;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
    }
    
    /* Expander content background */
    [data-testid="stSidebar"] .streamlit-expanderContent,
    [data-testid="stSidebar"] details[open] {
        background-color: rgba(58, 74, 94, 0.3) !important;
    }
    
    /* Dropdown menu items - oscuros */
    [data-testid="stSidebar"] [role="option"],
    [data-testid="stSidebar"] [role="listbox"],
    [data-testid="stSidebar"] ul[role="listbox"] li {
        background: linear-gradient(135deg, #3A4A5E 0%, #2C3A4E 100%) !important;
        color: #E8F1F5 !important;
    }
    
    [data-testid="stSidebar"] [role="option"]:hover {
        background: linear-gradient(135deg, #4A5A6E 0%, #3C4A5E 100%) !important;
        color: #00D4FF !important;
    }
    
    /* Slider styling - morado oscuro visible */
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #805AD5 !important;
    }
    
    [data-testid="stSidebar"] .stSlider [role="slider"] {
        background-color: #5B21B6 !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 2px 8px rgba(91, 33, 182, 0.4);
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: rgba(128, 90, 213, 0.3) !important;
        height: 6px !important;
    }
    
    /* Checkbox styling - blancos con borde cyan */
    [data-testid="stSidebar"] input[type="checkbox"] {
        width: 20px !important;
        height: 20px !important;
        border: 2px solid #00D4FF !important;
        background-color: #ffffff !important;
        border-radius: 4px;
        appearance: none;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] input[type="checkbox"]:checked {
        background-color: #00D4FF !important;
        border: 2px solid #00B4D8 !important;
    }
    
    [data-testid="stSidebar"] input[type="checkbox"]:checked::after {
        content: '✓';
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    
    [data-testid="stSidebar"] .stCheckbox {
        color: #E8F1F5 !important;
    }
    
    /* Number input styling */
    [data-testid="stSidebar"] input[type="number"],
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border: 2px solid #9F7AEA !important;
    }
    
    /* Generate button styling - Cyan gradient */
    .stButton > button,
    .stButton button,
    button[kind="primary"],
    button[kind="secondary"] {
        background: linear-gradient(135deg, #00D4FF 0%, #00B4D8 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover,
    .stButton button:hover {
        background: linear-gradient(135deg, #48CAE4 0%, #0096C7 100%) !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
        transform: translateY(-2px);
        color: #ffffff !important;
    }
    
    /* Headings in content - Alto contraste */
    h1 {
        color: #0F172A !important;
        font-weight: 800 !important;
        border-top: 3px solid #00D4FF;
        padding-top: 1rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    h2 {
        color: #1E293B !important;
        font-weight: 700 !important;
        border-top: 2px solid #9F7AEA;
        padding-top: 0.75rem;
    }
    
    h3 {
        color: #334155 !important;
        font-weight: 700 !important;
        border-top: 2px solid #9F7AEA;
        padding-top: 0.75rem;
    }
    
    /* Success messages - Light cyan con texto oscuro */
    .element-container .stSuccess {
        background: linear-gradient(135deg, #D1F4FF 0%, #B8E8F5 100%);
        border-left: 4px solid #00D4FF;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .element-container .stSuccess * {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    
    /* Warning messages - Light purple con texto oscuro */
    .element-container .stWarning {
        background: linear-gradient(135deg, #E9D8FD 0%, #D6BCFA 100%);
        border-left: 4px solid #9F7AEA;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .element-container .stWarning * {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    
    /* Info messages - Morado claro con texto oscuro */
    .element-container .stInfo {
        background: linear-gradient(135deg, #E9D8FD 0%, #D6BCFA 100%);
        border-left: 4px solid #9F7AEA;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .element-container .stInfo * {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    
    /* Text area styling - light with purple border */
    .stTextArea textarea {
        background: #ffffff !important;
        color: #1E3A5F !important;
        border: 2px solid #B794F6 !important;
        border-radius: 8px;
    }
    
    .stTextArea textarea:focus {
        border: 2px solid #9F7AEA !important;
        box-shadow: 0 0 8px rgba(159, 122, 234, 0.3);
    }
    
    /* Content paragraphs - alto contraste */
    p {
        color: #1E293B !important;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    
    /* Main content area cards/containers - light blue */
    .element-container {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Markdown content styling */
    .stMarkdown {
        color: #1E293B !important;
    }
    
    .stMarkdown strong {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #0F172A !important;
    }
    
    /* Link button styling - cyan */
    .stLinkButton > a {
        background: linear-gradient(135deg, #00D4FF 0%, #00B4D8 100%) !important;
        color: #ffffff !important;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 212, 255, 0.3);
    }
    
    .stLinkButton > a:hover {
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.5);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

def get_rag_engine():
    """Get or initialize RAG engine from session state"""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = ScienceRAG()
    return st.session_state.rag_engine

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
        topic_arxiv = st.text_input(
            "Tema de investigación", 
            value="", 
            placeholder="Ej: machine learning, drones, quantum computing",
            key="arxiv_topic_input"
        )
        num_papers = st.slider("Cantidad de papers (Máx 3 para evitar errores)", 1, 3, 1)
        
        if st.button("Actualizar Conocimiento"):
            if not topic_arxiv or not topic_arxiv.strip():
                st.error("Por favor, ingresa un tema de investigación")
            else:
                try:
                    with st.spinner("Descargando y procesando..."):
                        status = rag.ingest_papers(topic_arxiv.strip(), max_results=num_papers)
                    
                    if "Error" in status:
                        st.error(f"❌ {status}")
                        # No hacer rerun si hay error, para que se vea el mensaje
                    else:
                        st.success(f"✅ {status}")
                        st.cache_data.clear()
                        # Solo hacer rerun si fue exitoso
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al indexar papers: {str(e)}")
                    import traceback
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())
                    # No hacer rerun si hay excepción
            
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
            st.error("No hay documentos indexados.")
            
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
            "Tema del Contenido:",
            placeholder="Ej: Agujeros negros, inteligencia artificial, energía solar..."
        )
        audience = st.text_input(
            "Audiencia Objetivo:",
            placeholder="Ej: Todo el público, profesionales del sector, estudiantes..."
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
        rag = get_rag_engine()
        
        # Check if there are any papers indexed first
        indexed_papers = rag.list_indexed_papers()
        if not indexed_papers:
            st.warning("⚠️ No hay documentos indexados en la base de datos. Por favor, indexa papers primero desde el sidebar en 'Base de Datos Científica'.")
            documents = ""
            sources = []
        else:
            documents, sources = rag.get_context(topic)
        
        if indexed_papers and not documents:
            st.warning(f"No se encontraron documentos relevantes sobre '{topic}'. Hay {len(indexed_papers)} paper(s) indexados, pero no coinciden con tu tema. Intenta indexar papers específicos sobre '{topic}'.")
            
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
            
            # Display sources with enhanced UI
            if sources:
                st.markdown("---")
                st.markdown("### 📚 Fuentes Científicas (arXiv)")
                
                for idx, source in enumerate(sources, 1):
                    with st.expander(f"📄 **{idx}. {source['title']}**", expanded=False):
                        st.markdown(f"**👥 Autores:** {source.get('authors', 'Unknown')}")
                        st.markdown(f"**📅 Publicado:** {source.get('published', 'Unknown')}")
                        st.markdown(f"**🔗 Leer en arXiv:** [Ver Paper]({source['url']})")
                        st.link_button("📖 Abrir en arXiv", source['url'], use_container_width=True)

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
