from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import spacy
from datetime import datetime
import os
import wave
import json
import numpy as np
from pathlib import Path
import shutil

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temporary directory for audio files
TEMP_DIR = Path("temp_audio_files")
TEMP_DIR.mkdir(exist_ok=True)

# Load NLP model for text analysis
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class Summary(BaseModel):
    key_points: List[str]
    summary: str
    topics_discussed: List[str]
    duration: float
    interview_date: datetime
    transcript: str
    confidence_score: float
    speaker_count: Optional[int]

class AudioProcessor:
    """Class to handle audio processing with multiple backends"""
    
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Get audio duration using either wave or soundfile"""
        try:
            # Try soundfile first
            data, samplerate = sf.read(file_path)
            return len(data) / samplerate
        except Exception:
            try:
                # Fallback to wave
                with wave.open(file_path, 'rb') as audio_file:
                    frames = audio_file.getnframes()
                    rate = audio_file.getframerate()
                    return frames / float(rate)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not process audio file: {str(e)}"
                )

    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> None:
        """Convert audio file to WAV format"""
        try:
            data, samplerate = sf.read(input_path)
            sf.write(output_path, data, samplerate)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Audio conversion failed: {str(e)}"
            )

    @staticmethod
    def transcribe_audio(file_path: str) -> tuple[str, float]:
        """Transcribe audio file to text using multiple recognition attempts"""
        recognizer = sr.Recognizer()
        
        # Adjust recognition settings
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.15
        recognizer.dynamic_energy_ratio = 1.5
        recognizer.pause_threshold = 0.8
        
        text = ""
        confidence = 0.0
        
        try:
            with sr.AudioFile(file_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.record(source)
                
                # Try Google Speech Recognition first
                try:
                    result = recognizer.recognize_google(audio, show_all=True)
                    if result and 'alternative' in result:
                        text = result['alternative'][0]['transcript']
                        confidence = result['alternative'][0].get('confidence', 0.0)
                except:
                    # Fallback to other recognition methods if needed
                    text = recognizer.recognize_sphinx(audio) if text == "" else text
                    confidence = 0.5  # Default confidence for fallback method
                
                return text, confidence
                
        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Could not understand audio")
        except sr.RequestError:
            raise HTTPException(status_code=500, detail="Speech recognition service unavailable")

class TextAnalyzer:
    """Class to handle text analysis"""
    
    @staticmethod
    def extract_key_points(text: str) -> List[str]:
        """Extract key points from the text using NLP"""
        doc = nlp(text)
        key_sentences = []
        
        # Calculate sentence importance scores
        scores = {}
        for sent in doc.sents:
            score = sum([
                2 if ent.label_ in ['PERSON', 'ORG', 'SKILL', 'PRODUCT'] else 1
                for ent in sent.ents
            ]) + sum([
                1 for chunk in sent.noun_chunks
                if chunk.root.pos_ in ['VERB', 'NOUN']
            ])
            scores[sent] = score
        
        # Get top sentences
        sorted_sents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        key_sentences = [sent.text for sent, score in sorted_sents[:5]]
        
        return key_sentences

    @staticmethod
    def generate_summary(text: str) -> str:
        """Generate a concise summary of the interview"""
        doc = nlp(text)
        
        # Enhanced sentence scoring
        sentence_scores = {}
        for sent in doc.sents:
            score = sum([
                3 if ent.label_ in ['PERSON', 'ORG', 'SKILL', 'PRODUCT'] else 1
                for ent in sent.ents
            ]) + sum([
                1 for token in sent
                if not token.is_stop and token.pos_ in ['VERB', 'NOUN', 'ADJ']
            ])
            sentence_scores[sent.text] = score
        
        # Get top 3 sentences for summary
        summary_sentences = sorted(
            sentence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return " ".join([sent for sent, score in summary_sentences])

    @staticmethod
    def extract_topics(text: str) -> List[str]:
        """Extract main topics discussed in the interview"""
        doc = nlp(text)
        topics = {}
        
        # Enhanced topic extraction
        for ent in doc.ents:
            if ent.label_ in ['SKILL', 'PRODUCT', 'ORG', 'TOPIC', 'PERSON', 'EVENT']:
                # Normalize topic text
                topic_text = ent.text.lower()
                topics[topic_text] = topics.get(topic_text, 0) + 1
        
        # Add noun chunks that appear frequently
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Only short phrases
                topic_text = chunk.text.lower()
                if chunk.root.pos_ in ['NOUN', 'PROPN']:
                    topics[topic_text] = topics.get(topic_text, 0) + 1
        
        # Return top topics sorted by frequency
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, freq in sorted_topics[:5]]

    @staticmethod
    def estimate_speaker_count(text: str) -> int:
        """Estimate the number of speakers in the conversation"""
        doc = nlp(text)
        speaker_indicators = set()
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                speaker_indicators.add(ent.text.lower())
        
        # Look for speaker indicators like "I", "you", "they"
        pronouns = {'i', 'you', 'he', 'she', 'they'}
        for token in doc:
            if token.text.lower() in pronouns:
                speaker_indicators.add(token.text.lower())
        
        # Estimate based on unique indicators found
        if len(speaker_indicators) <= 1:
            return 1
        elif len(speaker_indicators) == 2:
            return 2
        else:
            return len(speaker_indicators)

class AudioAnalyzer:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.text_analyzer = TextAnalyzer()

    async def analyze_audio(self, file_path: str) -> Summary:
        """Complete audio analysis pipeline"""
        # Convert to WAV if needed
        file_ext = Path(file_path).suffix.lower()
        wav_path = file_path
        if file_ext != '.wav':
            wav_path = str(Path(file_path).with_suffix('.wav'))
            self.audio_processor.convert_to_wav(file_path, wav_path)

        # Get audio duration
        duration = self.audio_processor.get_audio_duration(wav_path)

        # Transcribe audio
        transcript, confidence = self.audio_processor.transcribe_audio(wav_path)

        # Analyze text
        key_points = self.text_analyzer.extract_key_points(transcript)
        summary = self.text_analyzer.generate_summary(transcript)
        topics = self.text_analyzer.extract_topics(transcript)
        speaker_count = self.text_analyzer.estimate_speaker_count(transcript)

        # Create response
        return Summary(
            key_points=key_points,
            summary=summary,
            topics_discussed=topics,
            duration=duration,
            interview_date=datetime.now(),
            transcript=transcript,
            confidence_score=confidence,
            speaker_count=speaker_count
        )

@app.post("/api/upload-audio", response_model=Summary)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Endpoint to upload audio file and get analysis
    """
    supported_formats = {'.wav', '.flac', '.ogg', '.mp3', '.m4a'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
        )
    
    # Create unique temporary file path
    temp_file = TEMP_DIR / f"temp_{datetime.now().timestamp()}{file_ext}"
    
    try:
        # Save uploaded file
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze audio
        analyzer = AudioAnalyzer()
        analysis = await analyzer.analyze_audio(str(temp_file))
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_files, temp_file)
        
        return analysis
    
    except Exception as e:
        # Cleanup on error
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_files(file_path: Path):
    """Clean up temporary files"""
    try:
        if file_path.exists():
            file_path.unlink()
        # Also remove converted WAV if it exists
        wav_path = file_path.with_suffix('.wav')
        if wav_path.exists():
            wav_path.unlink()
    except Exception as e:
        print(f"Error cleaning up files: {e}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "nlp_model": "loaded" if nlp is not None else "not_loaded",
        "temp_directory": str(TEMP_DIR),
        "supported_formats": [".wav", ".flac", ".ogg", ".mp3", ".m4a"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)