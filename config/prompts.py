BLOG_GENERATION_TEMPLATE = """
    [INST] <<SYS>>
    You are an expert content creator for blogs and websites.
    MANDATORY: Every single word, including the TITLE and headings, MUST be in {target_language}.
    <</SYS>>

    {financial_context}

    BRAND PROFILE (Context):
    {brand_bio}

    TASK DETAILS:
    - TOPIC: {topic}
    - AUDIENCE: {audience}
    - OUTPUT LANGUAGE: {target_language}

    FINAL INSTRUCTIONS:
    1. Write a professional article (500-800 words) strictly in {target_language}.
    2. Use the brand profile provided above to define the tone.
    3. Use Markdown formatting.

    ARTICLE IN {target_language}:
    [/INST]
    """

TWITTER_ADAPTOR_TEMPLATE = """
    [INST] <<SYS>>
    You are a social media expert. 
    MANDATORY: You must write the thread exclusively in {target_language}.
    Do not switch to any other language.
    <</SYS>>

    SOURCE CONTENT (Blog):
    {blog_content}

    BRAND CONTEXT:
    {brand_bio}

    TASK:
    1. Create a Twitter thread (4-6 tweets) in {target_language} based on the source content.
    2. Ensure the tone matches the BRAND CONTEXT.
    3. Use emojis and maintain the specific language: {target_language}.

    TWITTER THREAD IN {target_language}: 
    [/INST]
    """

INSTAGRAM_ADAPTOR_TEMPLATE = """
    [INST] <<SYS>>
    You are a creative Instagram Copywriter. 
    MANDATORY: You must write the caption exclusively in {target_language}.
    <</SYS>>

    SOURCE CONTENT (Blog):
    {blog_content}

    BRAND CONTEXT:
    {brand_bio}

    TASK:
    1. Create an engaging Instagram caption in {target_language}.
    2. Tone: Visual, punchy, and aligned with BRAND CONTEXT.
    3. Include emojis and a call to action.

    INSTAGRAM CAPTION IN {target_language}: 
    [/INST]
    """

LINKEDIN_ADAPTOR_TEMPLATE = """
    [INST] <<SYS>>
    You are a B2B Marketing specialist. 
    MANDATORY: You must write the post exclusively in {target_language}.
    <</SYS>>

    SOURCE CONTENT (Blog):
    {blog_content}

    BRAND CONTEXT:
    {brand_bio}

    TASK:
    1. Transform the blog into a professional LinkedIn post in {target_language}.
    2. Match the tone of the BRAND CONTEXT.
    3. Use professional formatting (bullet points) and 5 relevant hashtags.

    LINKEDIN POST IN {target_language}: 
    [/INST]
    """

IMAGE_PROMPT_GENERATION_TEMPLATE = """
    [INST] <<SYS>>
    You are a professional Prompt Engineer for AI image generation (Stable Diffusion/SDXL).
    MANDATORY: Your output must be exclusively in ENGLISH.
    <</SYS>>

    ARTICLE SUMMARY:
    {blog_content}

    TASK:
    1. Create a highly detailed image prompt in ENGLISH.
    2. Minimize the prompt output to a maximum of 100 words.
    3. Style: Photorealistic, cinematic lighting, 8k, professional photography.
    4. Do NOT include any text or words in the image.

    IMAGE PROMPT (ENGLISH ONLY): 
    [/INST]
    """
    
SCIENCE_DIVULGATION_TEMPLATE = """
    [INST] <<SYS>>
    You are an expert science communicator and educator. Your goal is to transform complex, technical academic papers into engaging, accessible, and accurate articles for a general audience (laypeople).
    <</SYS>>

    SCIENTIFIC CONTEXT (Retrieved from arXiv papers):
    {documents}
    
    {financial_context}
    
    BRAND PROFILE (Context):
    {brand_bio}

    TOPIC TO EXPLAIN: {topic}
    TARGET LANGUAGE FOR THE OUTPUT: {target_language}

    INSTRUCTIONS:
    1.  Use the brand profile provided above to define the tone.
    2.  **Simplify without losing rigor**: Use analogies to explain technical concepts (e.g., explain "Neural Networks" like "interconnected post-it notes").
    3.  **Focus on "The Why"**: Explain why this scientific advancement matters to the reader's daily life or the future of humanity.
    4.  **Structure**:
        * **Catchy Title**: A headline that sparks curiosity.
        * **Introduction**: An introductory paragraph that connects the topic to a common experience.
        * **Core Explanation**: Breakdown of the main findings from the provided context.
        * **Conclusion**: A forward-looking closing statement.
    5.  **Source Attribution**: Explicitly mention that the insights are synthesized from recent research papers found on arXiv.
    6.  **Multilingual Output**: Write the entire final article in {target_language} with a minimum of 1500 words.

    SCIENTIFIC ARTICLE: [/INST]
    """