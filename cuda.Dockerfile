FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /Project

# Copy requirements first to take advantage of Docker caching
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  libsndfile1 \
  ffmpeg \
  git \
  build-essential \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application files
COPY . .

# Create directory for temporary audio files
RUN mkdir -p temp_audio_files

# Command to run the app (you might want to adjust this based on which file you want to run)
CMD ["python3", "main.py"]
