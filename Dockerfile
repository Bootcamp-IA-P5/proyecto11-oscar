# --- STAGE 1: Base Image Build ---
# We use a lightweight Python 3.11/3.12 base image
FROM python:3.11-slim

# Set the timezone (useful for logs and consistency)
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create and set the working directory
WORKDIR /app

# Copy the requirements file and configurations
COPY requirements.txt .
COPY config/settings.py /app/config/settings.py

# Install dependencies
# Git is installed because some libraries (e.g., LangChain) may depend on it.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Execution Environment Preparation ---

# Copy the rest of the source code and the Streamlit application
COPY . /app

# Expose the default Streamlit port
EXPOSE 8501

# Entry command to run the Streamlit application
# We use CMD so it can be easily overridden if necessary
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]