from transformers import BartForConditionalGeneration, BartTokenizer
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import soundfile as sf
import whisper
import time
from pathlib import Path


# Load BART tokenizer and model
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)


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

    # Function to the audio duration
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        data, samplerate = sf.read(file_path)
        return len(data) / samplerate

    # Function to convert the audio files to .wav format if not already
    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> None:
        data, samplerate = sf.read(input_path)
        sf.write(output_path, data, samplerate)

    # Function to convert audio file to text
    @staticmethod
    def transcribe_audio(file_path: str) -> tuple[str, float]:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        transcript = result["text"]
        confidence = result.get("confidence", 0.0)
        return transcript, confidence


class TextAnalyzer:
    """Class to handle text analysis"""

    # Function to generate summary
    @staticmethod
    def generate_summary(text: str) -> str:
        inputs = bart_tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = bart_model.generate(inputs.input_ids, max_length=1000, min_length=40, length_penalty=2.0, num_beams=4)
        return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Function to extract topics from the text
    @staticmethod
    def extract_topics(text: str) -> List[str]:
        return ["Topic extraction is yet to be implemented"]

    # Function to ge the speaker count from the text
    @staticmethod
    def estimate_speaker_count(text: str) -> int:
        return 1

# The main function to anaylza the audio 
def analyze_audio(file_path: str) -> Summary:
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
