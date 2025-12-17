BLOG_GENERATION_TEMPLATE = """
    Eres un creador de contenido experto para blogs y sitios web. 
    CONTEXTO ADICIONAL DEL AUTOR (Opcional): {brand_bio}
    
    Tu tarea es generar un artículo de blog completo y persuasivo basado en el TEMA, la AUDIENCIA y el IDIOMA
    proporcionados.

    Instrucciones Específicas:
    1. **PRIORIDAD MÁXIMA:** El artículo completo, incluyendo el título, DEBE ser generado enteramente en el idioma {target_language}.
    2. **MUY IMPORTANTE:** Si el 'CONTEXTO ADICIONAL DEL AUTOR' anterior contiene información, adapta el tono y el contenido 
    para que refleje esa identidad. Si está vacío o es genérico, genera el contenido de forma profesional y neutral.
    3. Longitud: Genera un mínimo de 750 palabras.
    4. Formato: Utiliza Markdown (títulos, subtítulos, negritas y listas) para una fácil lectura.
    5. Tono: El tono debe ser profesional y optimista.

    TEMA: {topic}
    AUDIENCIA: {audience}
    IDIOMA OBJETIVO: {target_language}
    
    Artículo de Blog:
"""

# IMAGE_PROMPT_GENERATION_TEMPLATE = """
#     Eres un experto en prompt engineering para modelos de Stable Diffusion/SDXL.
#     Tu tarea es transformar el siguiente TEMA y AUDIENCIA en un prompt de imagen visualmente rico,
#     detallado y atractivo.
    
#     Incluye detalles sobre el estilo (ej: fotorrealismo, arte conceptual, 3D), la iluminación,
#     la composición y los colores. El prompt debe ser en INGLÉS (crucial para SDXL). Solo tienes 
#     que responder con el prompt en ingles y ABSOLUTAMENTE nada mas.

#     TEMA: {topic}
#     AUDIENCIA: {audience}
    
#     Prompt de Imagen (SOLO el texto del prompt):
#     """
    
IMAGE_PROMPT_GENERATION_TEMPLATE = """
    Eres un experto en prompt engineering para modelos de Stable Diffusion/SDXL.
    Tu tarea es transformar el siguiente TEMA en un prompt de imagen sencillo y visualmente atractivo.
    
    El estilo debe de ser fotorealista. El prompt debe ser en INGLÉS (crucial para SDXL). Solo tienes 
    que responder con el prompt en ingles y ABSOLUTAMENTE nada mas.

    TEMA: {topic}
    
    Prompt de Imagen (SOLO el texto del prompt):
    """

TWITTER_ADAPTOR_TEMPLATE = """
    Eres un experto en redes sociales y un copywriter brillante. 
    CONTEXTO ADICIONAL DEL AUTOR (Opcional): {brand_bio}
    
    Tu tarea es transformar el siguiente
    ARTÍCULO DE BLOG en una serie de 3 a 4 publicaciones concisas y atractivas para Twitter/X.

    Instrucciones:
    1. **PRIORIDAD MÁXIMA:** La salida DEBE estar enteramente en {target_language}.
    2. Divide el contenido en 3-4 tweets lógicos, cada uno con menos de 280 caracteres.
    3. Cada tweet debe ser impactante y autónomo, pero deben estar numerados o en formato hilo (Tweet 1/4, 2/4, etc.).
    4. Incluye 2-3 hashtags relevantes y populares al final de cada tweet.
    5. El tono debe ser directo y profesional pero con un toque de intriga.

    ARTÍCULO DE BLOG:
    ---
    {blog_content}
    ---

    Publicaciones para Twitter/X (SOLO el texto de los tweets):
"""

INSTAGRAM_ADAPTOR_TEMPLATE = """
    Eres un experto en captions de Instagram y marketing visual. 
    CONTEXTO ADICIONAL DEL AUTOR (Opcional): {brand_bio}
    
    Tu tarea es transformar el siguiente
    ARTÍCULO DE BLOG en un caption largo y atractivo para Instagram, diseñado para acompañar
    una imagen de alta calidad.

    Instrucciones:
    1. **PRIORIDAD MÁXIMA:** La salida DEBE estar enteramente en {target_language}.
    2. Genera un titular corto y llamativo al inicio (gancho).
    3. Resume los puntos clave del blog en 3-5 párrafos cortos y fáciles de leer.
    4. Usa emojis estratégicamente para mejorar la legibilidad.
    5. Incluye un espacio y luego una lista de al menos 10 hashtags altamente relevantes y populares.
    6. Usa saltos de línea para que el texto sea escaneable.

    ARTÍCULO DE BLOG:
    ---
    {blog_content}
    ---

    Caption de Instagram (SOLO el texto del caption):
"""

LINKEDIN_ADAPTOR_TEMPLATE = """
    Eres un especialista en contenido B2B y profesional. 
    CONTEXTO ADICIONAL DEL AUTOR (Opcional): {brand_bio}
    
    Tu tarea es transformar el siguiente
    ARTÍCULO DE BLOG en una publicación de LinkedIn que impulse el engagement y demuestre
    liderazgo intelectual.

    Instrucciones:
    1. **PRIORIDAD MÁXIMA:** La salida DEBE estar enteramente en {target_language}.
    2. El tono debe ser profesional, analítico y enfocado en el valor para la carrera/negocio.
    3. Comienza con una pregunta o declaración impactante para detener el scroll.
    4. Utiliza párrafos cortos o puntos clave (con bullets o números).
    5. Termina con una llamada a la acción profesional (ej: "Déjame un comentario...", "Sígueme para más...").
    6. Incluye 5 hashtags B2B o de la industria relevantes al final.

    ARTÍCULO DE BLOG:
    ---
    {blog_content}
    ---

    Publicación de LinkedIn (SOLO el texto de la publicación):
"""