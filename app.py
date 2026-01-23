"""
Streamlit app for intelligent content generation using LLM + RAG + Financial context.
"""

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
import io

from src.core.content_chains import (
    create_blog_chain,
    create_image_prompt_chain,
    create_instagram_adaptor_chain,
    create_linkedin_adaptor_chain,
    create_twitter_adaptor_chain,
    generate_science_post_chain,
    assemble_grounding_context,
    get_grounding_summary,
    _prepare_financial_context,
)


from src.models.image_generator import (
    generate_image_from_huggingface,
    generate_image_from_replicate,
    search_image_from_unsplash,
    search_image_from_pexels,
)
from src.core.rag_engine import ScienceRAG

from src.models.financial_news import (
    load_financial_news, 
    build_finance_context, 
    load_financial_news_multi,
    load_financial_news_newsapi
)


st.set_page_config(layout="wide")


# -----------------------
# Cached resources
# -----------------------

@st.cache_resource
def get_rag_engine():
    return ScienceRAG()


@st.cache_data(ttl=2)
def get_cached_papers(_rag_engine):
    return _rag_engine.list_indexed_papers()


# -----------------------
# Sidebar UI
# -----------------------

def render_sidebar():
    """Collects all UI inputs and returns them in a consistent order."""

    rag = get_rag_engine()
    st.sidebar.title("Generador de contenidos")

    
    # --- Financial News ---
    with st.sidebar.expander("📈 Noticias financieras"):

        finance_query = st.text_input(
            "Tema financiero",
            placeholder="Ej: inflation, ECB, Tesla, interest rates...",
            key="finance_query_input",
        )

        finance_max_articles = st.slider("Máx. artículos", 1, 10, 5)

        st.markdown("**Fuentes:**")
        use_alpha = st.checkbox("Alpha Vantage", value=False)
        use_newsapi = st.checkbox("NewsAPI", value=False)

        finance_enabled = use_alpha or use_newsapi



    # --- Scientific Knowledge Base (RAG) ---
    with st.sidebar.expander("🔬 Base de Datos Científica"):
        rag_enabled = st.checkbox(
            "Activar Base de Datos Científica (RAG)", value=True
        )
        topic_arxiv = st.text_input("Tema de investigación", value="LLM Safety")
        num_papers = st.slider("Cantidad de papers (Máx 3)", 1, 3, 1)

        if st.button("Actualizar Conocimiento"):
            with st.spinner("Descargando y procesando..."):
                rag.ingest_papers(topic_arxiv, max_results=num_papers)
                st.cache_data.clear()
                st.success("Conocimiento actualizado")

    # --- Available Documents ---
    with st.sidebar.expander("📄 Documentos disponibles"):
        docs = get_cached_papers(rag)

        if docs:
            st.write(f"Hay **{len(docs)}** documentos indexados:")
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"{i}. {doc}")

            if st.button("Limpiar Base de Datos"):
                rag.reset_database()
                st.cache_data.clear()
                st.warning("Base de datos borrada.")
                st.rerun()
        else:
            st.warning("No hay documentos indexados.")

    # --- Generation parameters ---
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

        brand_bio = st.text_area("Datos de la empresa/persona:")

        llm_display = st.selectbox("Seleccionar Proveedor", llm_options)
        llm_selection = llm_display.split("(")[0].strip()

        selected_language = st.selectbox(
            "Idioma a usar para la generación",
            list(language_map.keys()),
        )
        target_language = language_map[selected_language]

    # --- Main Inputs ---
    
    with st.sidebar:
        topic = st.text_input("Tema del Contenido:", "Agujeros negros")
        audience = st.text_input("Audiencia Objetivo:", "Todo el público")

        generate_twitter = st.checkbox("Adaptar a Twitter/X", value=False)
        generate_instagram = st.checkbox("Adaptar a Instagram", value=False)
        generate_linkedin = st.checkbox("Adaptar a LinkedIn", value=False)
        generate_image = st.checkbox("Generar imágenes", value=False)

        image_editorial = False
        image_creative = False

        if generate_image:
            image_editorial = st.checkbox("Imagen profesional (editorial)", value=True)
            image_creative = st.checkbox("Imagen creativa (marketing)", value=True)


        image_provider = None
        if generate_image:
            image_provider = st.selectbox(
                "Proveedor de Imagen",
                [
                    "Unsplash (Stock Photos)",
                    "Pexels (Stock Photos)",
                    "Hugging Face (SDXL)",
                    "Replicate (Flux)",
                ],
            )

        debug_mode = st.checkbox("🔧 Developer mode")
        generate_button = st.button("Generar Todo el Contenido", type="primary")

    return (
        brand_bio,
        llm_selection,
        target_language,
        topic,
        audience,
        rag_enabled,
        topic_arxiv,
        num_papers,
        finance_enabled,
        finance_query,
        finance_max_articles,
        use_alpha,
        use_newsapi,
        generate_twitter,
        generate_instagram,
        generate_linkedin,
        generate_image,
        image_editorial,
        image_creative,
        image_provider,
        debug_mode,
        generate_button,
    )


# -----------------------
# Core pipeline
# -----------------------

def generate_content(
    brand_bio,
    llm_selection,
    target_language,
    topic,
    audience,
    rag_enabled,
    topic_arxiv,
    num_papers,
    finance_enabled,
    finance_query,
    finance_max_articles,
    use_alpha,
    use_newsapi,
    generate_twitter,
    generate_instagram,
    generate_linkedin,
    generate_image,
    image_editorial,
    image_creative,
    image_provider,
    debug_mode,
):
    """Main orchestration logic."""

    blog_chain = create_blog_chain(llm_selection)
    science_chain = generate_science_post_chain(llm_selection)
    
    # -----------------------
    # Validaciones de entrada (UX y coherencia)
    # -----------------------

    # 1. Siempre exigir tema
    if not topic.strip():
        st.error("Debes indicar un tema para generar contenido.")
        return

    # 2. Finanzas activadas pero sin query → se desactiva suavemente
    if finance_enabled and not finance_query.strip():
        st.warning("Modo financiero activado pero sin tema financiero. Se ignorará el contexto financiero.")
        finance_enabled = False

    # 3. RAG activado pero sin documentos indexados
    if rag_enabled:
        rag = get_rag_engine()
        if not rag.list_indexed_papers():
            st.warning("RAG activado pero no hay documentos indexados. Pulsa 'Actualizar Conocimiento'.")


    # Financial context
    financial_context, financial_articles = "", []
    
    if finance_enabled and finance_query:

        if use_alpha and not use_newsapi:
            financial_articles = load_financial_news(
                query=finance_query,
                max_articles=finance_max_articles
            )

        elif use_newsapi and not use_alpha:
            financial_articles = load_financial_news_newsapi(
                query=finance_query,
                max_articles=finance_max_articles
            )

        else:
            # Both active → multi-source
            financial_articles = load_financial_news_multi(
                query=finance_query,
                max_articles=finance_max_articles
            )

    financial_context = build_finance_context(financial_articles)
    
    
    # RAG context
    documents = ""
    rag_sources = []
    rag_coverage = 0
    
    if rag_enabled:
        rag = get_rag_engine()     # SAME cached instance
        
        if rag.list_indexed_papers():
            rag_result = rag.get_context(topic, k_chunks=num_papers * 2)

            documents = rag_result["context"]
            rag_sources = rag_result["sources"]
            rag_coverage = rag_result["coverage"]
            
            # Heurística básica de relevancia
            if topic.lower() not in documents.lower():
                st.warning("El conocimiento científico recuperado no parece relevante para el tema. Se usará generación estándar.")
                documents = ""
                rag_sources = []
                rag_coverage = 0
            
            if debug_mode:
                st.write("DEBUG - Documentos recuperados:", documents[:500])
                
    # Only warn when financial context is being used as secondary (not main topic)
    # Finance is background if RAG dominates and topics differ
    if finance_enabled and financial_articles and rag_enabled and documents:
        if topic.lower() not in finance_query.lower():
            financial_context = (
                "NOTE: Financial articles provided are background context only. "
                "Prioritize topic relevance.\n\n" + financial_context
            )

            st.info(
                "ℹ️ Las noticias financieras se han usado solo como contexto general "
                "(no están directamente relacionadas con el tema principal)."
            )
            
    # Stop generation if only financial mode is active and no articles were retrieved
    if finance_enabled and not rag_enabled and not financial_articles:
        st.error(
            "No se han podido recuperar noticias financieras para esa búsqueda. "
            "Prueba con otro término (ej: inflation, Tesla, ECB, markets)."
        )
        return
    
    combined_context = assemble_grounding_context(
        rag_context=documents if rag_enabled else "",
        financial_context=financial_context if finance_enabled else "",
        debug=debug_mode,
    )
    
    
    # -----------------------
    # Transparency layer (user + developer feedback)
    # -----------------------

    active_modes = []

    if rag_enabled:
        active_modes.append("🔬 Científico (RAG)")

    if finance_enabled:
        active_modes.append("📈 Financiero")

    if not active_modes:
        active_modes.append("🧠 LLM puro")

    st.info(f"Modo activo: {' + '.join(active_modes)}")
    
    
    # Warnings informativos
    if rag_enabled and not documents:
        st.warning("RAG activado pero sin contexto relevante. Se utilizó generación LLM estándar.")

    if finance_enabled and finance_query and not financial_articles:
        st.warning("Noticias financieras activadas pero no se recuperaron artículos.")
        
    
    
    # Matrix logic respected
    if rag_enabled:
        blog_content = science_chain.invoke({
            "documents": documents,
            "topic": topic,
            "target_language": target_language,
            "brand_bio": brand_bio,
            "financial_context": financial_context,
        })
    else:
        blog_content = blog_chain.invoke({
            "topic": topic,
            "audience": audience,
            "target_language": target_language,
            "brand_bio": brand_bio,
            "financial_context": financial_context,
        })

    st.markdown("### 📝 Artículo de Blog")
    st.markdown(blog_content)
    
    # -----------------------
    # PDF Generation
    # -----------------------
    
    st.divider()

if st.button("📄 Exportar a PDF"):
    pdf = generate_pdf(
        blog_content=blog_content,
        twitter=twitter_post if generate_twitter else None,
        instagram=instagram_post if generate_instagram else None,
        linkedin=linkedin_post if generate_linkedin else None,
        rag_sources=rag_sources if rag_enabled else None,
        financial_articles=financial_articles if finance_enabled else None
    )

    st.download_button(
        label="Descargar PDF",
        data=pdf,
        file_name="nemotecas_ai_content.pdf",
        mime="application/pdf"
    )

    
    
    # -----------------------
    # Social Media Adaptations
    # -----------------------

    if generate_twitter:
        twitter_chain = create_twitter_adaptor_chain(llm_selection)
        with st.spinner("Generando hilo para Twitter/X..."):
            twitter_post = twitter_chain.invoke({
                "blog_content": blog_content,
                "brand_bio": brand_bio,
                "target_language": target_language,
            })

        st.markdown("### 🐦 Twitter/X")
        st.markdown(twitter_post)


    if generate_instagram:
        instagram_chain = create_instagram_adaptor_chain(llm_selection)
        with st.spinner("Generando caption para Instagram..."):
            instagram_post = instagram_chain.invoke({
                "blog_content": blog_content,
                "brand_bio": brand_bio,
                "target_language": target_language,
            })

        st.markdown("### 📸 Instagram")
        st.markdown(instagram_post)


    if generate_linkedin:
        linkedin_chain = create_linkedin_adaptor_chain(llm_selection)
        with st.spinner("Generando post para LinkedIn..."):
            linkedin_post = linkedin_chain.invoke({
                "blog_content": blog_content,
                "brand_bio": brand_bio,
                "target_language": target_language,
            })

        st.markdown("### 💼 LinkedIn")
        st.markdown(linkedin_post)

    # -----------------------
    # Image generation
    # -----------------------

    if generate_image:
        image_prompt_chain = create_image_prompt_chain(llm_selection)

        with st.spinner("Generando prompts de imagen..."):
            img_output = image_prompt_chain.invoke({
                "blog_content": blog_content
            })

        if debug_mode:
            st.text("DEBUG - Image prompts raw output:")
            st.text(img_output)

        # Parse prompts
        editorial_prompt = ""
        creative_prompt = ""

        try:
            parts = img_output.split("[CREATIVE PROMPT]")
            editorial_prompt = parts[0].split("[EDITORIAL PROMPT]")[1].strip()
            creative_prompt = parts[1].strip()
        except Exception:
            st.error("No se pudieron separar correctamente los prompts de imagen.")
            return

        # Generate selected images
        if image_provider:

            if image_editorial:
                st.markdown("### 🖼️ Imagen profesional")
                with st.spinner(f"Generando imagen editorial con {image_provider}..."):
                    if "Replicate" in image_provider:
                        img = generate_image_from_replicate(editorial_prompt)
                    else:
                        img = generate_image_from_huggingface(editorial_prompt)

                    if img:
                        st.image(img, use_container_width=True)

            if image_creative:
                st.markdown("### 🎨 Imagen creativa")
                with st.spinner(f"Generando imagen creativa con {image_provider}..."):
                    if "Replicate" in image_provider:
                        img = generate_image_from_replicate(creative_prompt)
                    else:
                        img = generate_image_from_huggingface(creative_prompt)

                    if img:
                        st.image(img, use_container_width=True)



    # -----------------------
    # Academic financial evidence layer (UI)
    # -----------------------

    if finance_enabled and financial_articles:
        st.markdown("### 📈 Artículos financieros utilizados")

        for art in financial_articles:
            provider = art.get("provider", "unknown")
            st.markdown(
                f"- **{art['title']}**  \n"
                f"  📰 {art['source']}  \n"
                f"  🌐 Fuente: `{provider}`  \n"
                f"  🔗 {art['url']}"
            )
    
    
    # -----------------------
    # Academic evidence layer (RAG)
    # -----------------------
    
    if rag_enabled and rag_sources:
        st.markdown("### 📄 Papers utilizados")

        for src in rag_sources:
            if src["url"]:
                st.markdown(
                    f"- **{src['title']}**  \n"
                    f"  🔗 {src['url']}  \n"
                    f"  Fragmentos usados: {src['chunks']}"
                )
            else:
                st.markdown(f"- **{src['title']}** (sin enlace disponible)")

    if rag_enabled:
        st.markdown(f"🔍 **Cobertura científica (RAG):** {rag_coverage}%")
    elif finance_enabled:
        st.markdown("🔍 **Cobertura científica (RAG):** N/A (modo financiero)")


def generate_pdf(
    blog_content,
    logo_path="assets/logo.png",
    twitter=None,
    instagram=None,
    linkedin=None,
    rag_sources=None,
    financial_articles=None
):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        name="Title",
        fontSize=20,
        spaceAfter=20,
        alignment=TA_CENTER
    )

    normal = styles["BodyText"]

    footer_style = ParagraphStyle(
        name="Footer",
        fontSize=9,
        textColor="grey",
        alignment=TA_CENTER
    )

    elements = []

    # Logo
    try:
        elements.append(Image(logo_path, width=200, height=80))
        elements.append(Spacer(1, 20))
    except:
        pass  # si no encuentra logo no rompe

    # Title (extract first markdown header or use default)
    elements.append(Paragraph("Generated Content – Nemotecas AI", title_style))
    elements.append(Spacer(1, 10))

    # Blog
    elements.append(Paragraph("<b>Blog Article</b>", styles["Heading2"]))
    for paragraph in blog_content.split("\n\n"):
        elements.append(Paragraph(paragraph.replace("\n", "<br/>"), normal))
        elements.append(Spacer(1, 8))

    # RRSS
    if twitter:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>Twitter/X</b>", styles["Heading2"]))
        elements.append(Paragraph(twitter.replace("\n", "<br/>"), normal))

    if instagram:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>Instagram</b>", styles["Heading2"]))
        elements.append(Paragraph(instagram.replace("\n", "<br/>"), normal))

    if linkedin:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>LinkedIn</b>", styles["Heading2"]))
        elements.append(Paragraph(linkedin.replace("\n", "<br/>"), normal))

    # RAG sources
    if rag_sources:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>Scientific Sources (RAG)</b>", styles["Heading2"]))
        for src in rag_sources:
            line = f"{src['title']} - {src.get('url', '')}"
            elements.append(Paragraph(line, normal))

    # Financial sources
    if financial_articles:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>Financial Sources</b>", styles["Heading2"]))
        for art in financial_articles:
            line = f"{art['title']} - {art['url']}"
            elements.append(Paragraph(line, normal))

    # Footer disclaimer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        "Nemotecas AI can make mistakes. Check important info.",
        footer_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# -----------------------
# Main
# -----------------------

def main():
    st.subheader("Contenido Generado (PoC)")
    values = render_sidebar()

    if values[-1]:
        generate_content(*values[:-1])


if __name__ == "__main__":
    main()
