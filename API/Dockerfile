FROM python:3.11-slim

# Set working directory
WORKDIR /Project

# Copy only the requirements.txt first (to take advantage of Docker caching)
COPY requirements.txt .

# Install system dependencies for audio processing, and other required libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support (ensure you're using compatible CUDA version with your environment)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Download the spaCy model (and potentially add more models if necessary)
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application files into the container
COPY . .

# Create a directory for temporary audio files if it's required by your code
RUN mkdir -p /Project/temp_audio_files

# Command to run the app (adjust if your entry point is different)
CMD ["python3", "main.py"]
