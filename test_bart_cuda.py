from transformers import BartForConditionalGeneration, BartTokenizer
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
from pathlib import Path
import soundfile as sf
import whisper
import torch
import time

# Load BART tokenizer and model
bart_model_name = "facebook/bart-large-cnn"  # Pretrained BART model for summarization
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to("cuda")  # Move BART model to GPU


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
        """Get audio duration using soundfile"""
        try:
            data, samplerate = sf.read(file_path)
            return len(data) / samplerate
        except Exception as e:
            raise ValueError(f"Could not process audio file: {str(e)}")

    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> None:
        """Convert audio file to WAV format"""
        try:
            data, samplerate = sf.read(input_path)
            sf.write(output_path, data, samplerate)
        except Exception as e:
            raise ValueError(f"Audio conversion failed: {str(e)}")

    @staticmethod
    def transcribe_audio(file_path: str) -> tuple[str, float]:
        """Transcribe audio file to text using OpenAI Whisper"""
        try:
            model = whisper.load_model("base", device="cuda")  # Load Whisper model on GPU
            result = model.transcribe(file_path)
            transcript = result["text"]
            confidence = result.get("confidence", 0.0)  # Whisper may not have confidence scores
            return transcript, confidence
        except Exception as e:
            raise RuntimeError(f"Audio transcription failed: {str(e)}")


class TextAnalyzer:
    """Class to handle text analysis"""

    @staticmethod
    def generate_summary(text: str) -> str:
        """Generate a concise summary using BART model"""
        inputs = bart_tokenizer(text, max_length=1024, return_tensors="pt", truncation=True).to("cuda")  # Move inputs to GPU
        summary_ids = bart_model.generate(
            inputs.input_ids,
            max_length=1000,
            min_length=40,
            length_penalty=2.0,
            num_beams=4
        )
        return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    @staticmethod
    def extract_topics(text: str) -> List[str]:
        """Placeholder for topic extraction"""
        return ["Topic extraction is yet to be implemented"]

    @staticmethod
    def estimate_speaker_count(text: str) -> int:
        """Placeholder for speaker count estimation"""
        return 1


def analyze_audio(file_path: str) -> Summary:
    """Complete audio analysis pipeline"""
    file_ext = Path(file_path).suffix.lower()
    wav_path = file_path
    if file_ext != '.wav':
        wav_path = str(Path(file_path).with_suffix('.wav'))
        AudioProcessor.convert_to_wav(file_path, wav_path)

    duration = AudioProcessor.get_audio_duration(wav_path)
    transcript, confidence = AudioProcessor.transcribe_audio(wav_path)
    summary = TextAnalyzer.generate_summary(transcript)
    topics = TextAnalyzer.extract_topics(transcript)
    speaker_count = TextAnalyzer.estimate_speaker_count(transcript)

    return Summary(
        key_points=[],
        summary=summary,
        topics_discussed=topics,
        duration=duration,
        interview_date=datetime.now(),
        transcript=transcript,
        confidence_score=confidence,
        speaker_count=speaker_count
    )


def test_analyze_audio():
    start_time = time.time()
    audio_file_path = "some_audio.wav"
    try:
        summary = analyze_audio(audio_file_path)
        print("Summary:", summary.summary)
        print("Topics Discussed:", summary.topics_discussed)
        print("Duration:", summary.duration)
        print("Interview Date:", summary.interview_date)
        print("Transcript:", summary.transcript)
        print("Confidence Score:", summary.confidence_score)
        print("Speaker Count:", summary.speaker_count)
    except Exception as e:
        print("Error:", e)
    end_time = time.time()
    print(f"time taken: {end_time - start_time}")


if __name__ == "__main__":
    test_analyze_audio()
