# Content Generator

## 🚀 Empowering Your Content Creation with AI-Driven Multi-Platform Adaptation

This project is an advanced, AI-powered content generation system designed to streamline and automate the creation of engaging content across multiple digital platforms. Leveraging the power of Large Language Models (LLMs), it allows users to generate comprehensive blog articles and then seamlessly adapt them for various social media channels like Twitter/X, Instagram, and LinkedIn. Additionally, it can generate compelling cover images to accompany your articles.

### Core Purpose
The primary goal is to assist content creators, marketers, social media managers, and bloggers in rapidly producing high-quality, platform-optimized content, reducing manual effort and increasing consistency across their digital presence.

### Target Audience
Content creators, digital marketers, social media managers, bloggers, small businesses, and anyone looking to enhance their content production workflow with AI.

### Main Functionalities
The application provides a user-friendly Streamlit interface to interact with powerful LLMs, enabling the generation and adaptation of content based on user-defined topics and target audiences.

### Key Features
*   **Increase knowledge:** Search and download technical papers from Arxiv.
*   **Multilingual:** Choose the output language
*   **Personalization:** Personalize the generation of the documents with your personal/company information.
*   **Blog Article Generation:** Create detailed and structured blog posts from a given topic and audience.
*   **Multi-Platform Content Adaptation:**
    *   **Twitter/X:** Condense and reformat blog content into concise, engaging tweets.
    *   **Instagram:** Generate compelling captions suitable for Instagram posts.
    *   **LinkedIn:** Craft professional and insightful posts for LinkedIn.
*   **AI-Powered Image Generation:** Automatically generate relevant cover images for articles based on the content's topic.
*   **Flexible LLM Provider Selection:** Choose between various LLM providers (e.g., Gemini, Groq, Ollama) to suit your performance and deployment needs.
*   **Interactive Streamlit UI:** A simple and intuitive web interface for easy content generation and management.

### Thecnologies used
* LangChain. The framework to connect to AI agents:
    * Groq
    * Gemini
    * Ollama
    * HuggingFace Hub
* LangSmith. To trace the usage of the agents (requests and responses)
* Replicate. To easy access to LLM models (only used for image generation)
* ChomaDb. To store and reuse the downloaded papers:
    * Arxiv
    * PyPdf
    * Sentence Transformers.
* Streamlit. To put everything together in a web page.

## 🏁 Getting Started

To get a copy of the project up and running on your local machine, follow these simple steps.

### Prerequisites
Ensure you have the following software installed on your system:

*   **Git:** Latest stable version.
    *   [Download Git](https://git-scm.com/downloads)
*   **Python:** Version 3.11 or 3.12.
    *   [Download Python](https://www.python.org/downloads/)
*   **pip:** Python package installer (usually comes with Python).
    *   `python -m ensurepip --upgrade`
*   **Docker:** Latest stable version, including Docker Buildx for multi-architecture builds.
    *   Download Docker Desktop
*   **(Optional) Ollama:** For local LLM inference if you choose the "Ollama" provider.
    *   Install Ollama

## 💻 Environment Setup (Local)

Follow these instructions to set up the project for local development. As an option, the project is also ready to be used with dev containers if you use vscode as your IDE. You can safely skip steps 2 and 3 if you choose to use dev containers for developemnt (the choice is up to you).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/content-generator.git # Replace with your actual repository URL
    cd content-generator
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables (if any):**
    *   If your project requires API keys or other sensitive information, create a `.env` file in the root directory based on a `.env.example` (if provided).
    *   For example, you might need to set `HUGGINGFACE_API_KEY` for image generation or specific LLM provider keys.

5.  **Ollama Setup (Optional - for local LLM inference):**
    If you plan to use Ollama as your LLM provider, ensure it's running locally and you have the necessary models pulled.
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull mistral # Or any other model you wish to use
    ```
    Ensure the Ollama server is running on `http://localhost:11434`.

6.  **Run the application in Local Development Mode:**
    ```bash
    streamlit run app.py
    ```
    The application will typically open in your web browser at `http://localhost:8501`.

## 🐳 Running with Docker (Recommended)

Docker provides a consistent and isolated environment for running the application.

### Building Docker Images

You can build a single-architecture Docker image or use the provided script to build multi-architecture images and push them to Docker Hub.

1.  **Build a single-architecture image (e.g., for your current machine):**
    ```bash
    docker build -t your-dockerhub-username/content-generator:latest .
    ```
    Replace `your-dockerhub-username` with your Docker Hub ID.

2.  **Build and Push Multi-Architecture Images (Recommended for public repositories):**
    This project includes a `build_and_push.sh` script that leverages Docker Buildx to create images for `linux/amd64` and `linux/arm64` and push them to your Docker Hub repository.

    *   **Prerequisites:** Ensure you are logged into Docker Hub from your terminal:
        ```bash
        docker login
        ```
    *   **Make the script executable:**
        ```bash
        chmod +x build_and_push.sh
        ```
    *   **Run the script:**
        ```bash
        ./build_and_push.sh your-dockerhub-username
        ```
        Replace `your-dockerhub-username` with your Docker Hub ID. This command will build and push the images.

### Running the Application with Docker

Once the Docker image is built (or pulled from Docker Hub), you can run the application:

```bash
docker run -p 8501:8501 your-dockerhub-username/content-generator:latest
```
This command maps port `8501` from the container to port `8501` on your host machine.

## 💡 Usage

After starting the application (either locally or via Docker), open your web browser and navigate to `http://localhost:8501`.

1.  **Technical database:** Search for technical papers and download them.
2.  **Browse the database:** List the downloaded papers.
3.  **Input Parameters:** Tune up the content generation: enter your personal information, choose the default LLM or select the output language.
4.  **Topic and audience:** Enter the `topic` for your content and define your `audience`.
5.  **Select Adaptations:** Check the boxes for the platforms you want to adapt the content for (Twitter/X, Instagram, LinkedIn) and if you want to generate a cover image.
6.  **Generate Content:** Click the "Generar Todo el Contenido" button.
7.  **View Results:** The generated blog article and its adaptations for selected platforms, along with the cover image, will appear in the main content area.

## 🤝 Contributing & Support

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

*   **Contributing:**
    1.  Fork the Project
    2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
    3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
    4.  Push to the Branch (`git push origin feature/AmazingFeature`)
    5.  Open a Pull Request

*   **Reporting Issues:**
    If you encounter any bugs or have feature requests, please open an issue on the GitHub Issues page.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
