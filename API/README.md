# Audio Summarization API
This repository contains a Flask-based code for processing audio files. It transcribes audio, summarizes the text, and sends back the summary along with other metadata.

## Features
- Receives audio files via an API endpoint.
- Converts audio to text using OpenAI Whisper.
- Summarizes the transcript using Facebook's BART model.
- Returns detailed metadata, including:
  - Transcript
  - Summary
  - Topics discussed
  - Duration
  - Confidence score
  - Speaker count (placeholder for now)
 
## Setup
1. Clone the Repository
```bash
git clone <repo URL>
cd <repo_name>
```
2. Install Python and Virtual Environment
- Ensure Python 3.8+ is installed on your system. It's recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
3. Install Dependencies
- Install all required dependencies from `requirements.txt`
```bash
pip install -r requirements.txt
```
4. Run the Flask Server
- Start the Flask server:
```bash
python app.py
```
- The server will start running at `http://127.0.0.1:5000`.

## How to Use
API Endpoint `/api/process_audio`
Method: `POST`
Request:
  - **File**: Upload an audio file (`.mp3` or `.wav`).
Response:
Returns a JSON object with:
  - `summary`: The generated summary.
  - `topics_discussed`: Topics extracted from the transcript.
  - `duration`: Duration of the audio file (seconds).
  - `transcript`: Full transcription of the audio.
  - `confidence_score`: Confidence level of the transcription.
  - `speaker_count`: Estimated number of speakers (placeholder).
Example (Using `curl`):
```bash
curl -X POST -F "audio_file=@example.mp3" http://127.0.0.1:5000/api/process_audio
```

## Using Docker
1. Building the Docker Image
```bash
docker build -t audio-summarizer .
```
2. Run the Docker Container
```bash
docker run -p 5000:5000 -v $(pwd)/uploads:/app/uploads audio-summarizer
```
  - Maps port `5000` of the container to your host.
  - Mounts the `uploads` directory for temporary file storage.
3. Test the API
Once the container is running, test the API at `http://127.0.0.1:5000/api/process_audio` as described above.

## Testing
To test the audio processing pipeline, run the provided `test_bart.py` script:
```bash
python test_bart.py
```
Ensure you have audio file (e.g. `some_audio.mp3`) in the root directory before running the script. 

## File Structure
```graphpl
project/
│
├── app.py                 # Flask API for audio processing
├── test_bart.py           # Script for testing the summarization pipeline
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker configuration for the project
├── uploads/               # Temporary folder for audio files
└── README.md              # Documentation
```

## Dependencies
The project uses the following libraries:
- Flask
- Transformers
- Torch
- OpenAI Whisper
- SoundFile
- Pydantic
All dependencies are listed in `requirements.txt` and are installed automatically.
