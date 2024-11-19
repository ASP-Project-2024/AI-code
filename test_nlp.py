from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
from pathlib import Path
import soundfile as sf
import spacy
import whisper
import time

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
            model = whisper.load_model("base")  # Choose model size; "base" is faster
            result = model.transcribe(file_path)
            transcript = result["text"]
            confidence = result.get("confidence", 0.0)  # Whisper models may not have confidence scores
            return transcript, confidence
        except Exception as e:
            raise RuntimeError(f"Audio transcription failed: {str(e)}")

class TextAnalyzer:
    """Class to handle text analysis"""
    
    @staticmethod
    def extract_key_points(text: str) -> List[str]:
        """Extract key points from the text using NLP"""
        doc = nlp(text)
        key_sentences = []
        
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
        
        sorted_sents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        key_sentences = [sent.text for sent, score in sorted_sents[:5]]
        
        return key_sentences

    @staticmethod
    def generate_summary(text: str) -> str:
        """Generate a concise summary of the interview"""
        doc = nlp(text)
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
        
        for ent in doc.ents:
            if ent.label_ in ['SKILL', 'PRODUCT', 'ORG', 'TOPIC', 'PERSON', 'EVENT']:
                topic_text = ent.text.lower()
                topics[topic_text] = topics.get(topic_text, 0) + 1
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                topic_text = chunk.text.lower()
                if chunk.root.pos_ in ['NOUN', 'PROPN']:
                    topics[topic_text] = topics.get(topic_text, 0) + 1
        
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
        
        pronouns = {'i', 'you', 'he', 'she', 'they'}
        for token in doc:
            if token.text.lower() in pronouns:
                speaker_indicators.add(token.text.lower())
        
        return max(len(speaker_indicators), 1)

def analyze_audio(file_path: str) -> Summary:
    """Complete audio analysis pipeline"""
    file_ext = Path(file_path).suffix.lower()
    wav_path = file_path
    if file_ext != '.wav':
        wav_path = str(Path(file_path).with_suffix('.wav'))
        AudioProcessor.convert_to_wav(file_path, wav_path)

    duration = AudioProcessor.get_audio_duration(wav_path)
    transcript, confidence = AudioProcessor.transcribe_audio(wav_path)
    key_points = TextAnalyzer.extract_key_points(transcript)
    summary = TextAnalyzer.generate_summary(transcript)
    topics = TextAnalyzer.extract_topics(transcript)
    speaker_count = TextAnalyzer.estimate_speaker_count(transcript)

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
    print(f"time taken: {end_time-start_time}")


if __name__ == "__main__":
    test_analyze_audio()
