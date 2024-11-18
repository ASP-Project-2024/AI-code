FROM python:3.11-slim

# Set working directory
WORKDIR /Project

# Copy requirements first to take advantage of Docker caching
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Command to run the app
CMD ["python3", "test_bart.py"]