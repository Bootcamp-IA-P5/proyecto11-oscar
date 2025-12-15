BLOG_GENERATION_TEMPLATE = """
Eres un experto en marketing de contenidos. Tu tarea es generar un artículo de blog completo 
y atractivo.

Instrucciones:
1. Genera un título pegadizo para el blog (usa encabezados de Markdown).
2. Escribe una introducción atractiva.
3. Desarrolla 3-4 puntos clave en secciones separadas (usa encabezados de Markdown).
4. Concluye con un resumen y una llamada a la acción clara.
5. El contenido debe estar formateado en Markdown sin iconos.

Tema a desarrollar: {topic}
Audiencia principal: {audience}
"""

IMAGE_PROMPT_GENERATION_TEMPLATE = """
    Eres un experto en prompt engineering para modelos de Stable Diffusion/SDXL.
    Tu tarea es transformar el siguiente TEMA y AUDIENCIA en un prompt de imagen visualmente rico,
    detallado y atractivo.
    
    Incluye detalles sobre el estilo (ej: fotorrealismo, arte conceptual, 3D), la iluminación,
    la composición y los colores. El prompt debe ser en INGLÉS (crucial para SDXL).

    TEMA: {topic}
    AUDIENCIA: {audience}
    
    Prompt de Imagen (SOLO el texto del prompt):
    """

TWITTER_ADAPTOR_TEMPLATE = """
Eres un experto en redes sociales y un copywriter brillante. Tu tarea es transformar el siguiente
ARTÍCULO DE BLOG en una serie de 3 a 4 publicaciones concisas y atractivas para Twitter/X.

Instrucciones:
1. Divide el contenido en 3-4 tweets lógicos, cada uno con menos de 280 caracteres.
2. Cada tweet debe ser impactante y autónomo, pero deben estar numerados o en formato hilo (Tweet 1/4, 2/4, etc.).
3. Incluye 2-3 hashtags relevantes y populares al final de cada tweet.
4. El tono debe ser directo y profesional pero con un toque de intriga.

ARTÍCULO DE BLOG:
---
{blog_content}
---

Publicaciones para Twitter/X (SOLO el texto de los tweets):
"""