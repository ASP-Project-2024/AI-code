from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import soundfile as sf
import whisper
import torch
from pathlib import Path
from pydub import AudioSegment
import re
import openai

# openai.api_key = "API_KEY_HERE"


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
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        data, samplerate = sf.read(file_path)
        return len(data) / samplerate

    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> None:
        try:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
        except Exception as e:
            raise ValueError(f"Audio conversion failed: {str(e)}")

    @staticmethod
    def transcribe_audio(file_path: str) -> tuple[str, float]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base").to(device)
        result = model.transcribe(file_path)
        transcript = result["text"]
        confidence = result.get("confidence", 0.0)
        return transcript, confidence


class TextAnalyzer:
    @staticmethod
    def detect_questions_and_answers(transcript: str) -> List[str]:
        question_pattern = r"(?:^|[\.\?!]\s)(Who|What|When|Where|Why|How|Is|Are|Do|Does|Can|Could|Would|Should|Did)\b.*?\?"
        questions = re.findall(question_pattern, transcript, flags=re.IGNORECASE)
        sentences = re.split(r"(?<=[.!?])\s+", transcript)

        qa_pairs = []
        question = None
        for sentence in sentences:
            if re.match(question_pattern, sentence, flags=re.IGNORECASE):
                question = sentence.strip()
            elif question:
                qa_pairs.append(f"Q: {question}\nA: {sentence.strip()}")
                question = None

        return qa_pairs

    @staticmethod
    def generate_summary(text: str) -> str:
        """
        Generate a concise summary using OpenAI ChatGPT.
        """
        try:
            prompt = (
                "You are a highly skilled assistant helping to summarize interview transcripts. "
                "The summary should be structured as follows:\n"
                "1. **Key Points**: List around 1-10 important points (if any) talked about in the transcript. If the transcript is small, you can list fewer important points.\n"
                "2. **Key Questions Asked**: List the main questions asked during the interview.\n"
                "3. **Candidate's Responses**: Summarize the main points of the candidate's answers.\n"
                "4. **Key Strengths or Skills Identified**: Highlight any specific strengths or skills the candidate mentioned.\n"
                "5. **Follow-Up Topics**: List any unresolved points or topics that might need further discussion.\n\n"
                "Here is the transcript of the interview:\n"
                f"{text}\n\n"
                "Please follow the structure and keep the summary clear and professional."
            )
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo"
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specialized in summarizing text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=500,  # Adjust based on the summary length
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Error generating summary: {str(e)}")

    @staticmethod
    def extract_topics(text: str) -> List[str]:
        return ["Topic extraction is yet to be implemented"]

    @staticmethod
    def estimate_speaker_count(text: str) -> int:
        return 1


def analyze_audio(file_path: str) -> Summary:
    file_ext = Path(file_path).suffix.lower()
    wav_path = file_path
    if file_ext != ".wav":
        wav_path = str(Path(file_path).with_suffix(".wav"))
        AudioProcessor.convert_to_wav(file_path, wav_path)

    duration = AudioProcessor.get_audio_duration(wav_path)
    transcript, confidence = AudioProcessor.transcribe_audio(wav_path)
    chatgpt_summary = TextAnalyzer.generate_summary(transcript)
    topics = TextAnalyzer.extract_topics(transcript)
    speaker_count = TextAnalyzer.estimate_speaker_count(transcript)

    key_points = []
    if chatgpt_summary:
        key_points_section = re.search(
            r"1\.\s*\*\*Key Points\*\*:(.*?)\n", chatgpt_summary, re.DOTALL
        )
        if key_points_section:
            key_points_text = key_points_section.group(1).strip()
            key_points = [
                point.strip() for point in key_points_text.split("\n") if point.strip()
            ]

    return Summary(
        key_points=key_points,
        summary=chatgpt_summary,
        topics_discussed=topics,
        duration=duration,
        interview_date=datetime.now(),
        transcript=transcript,
        confidence_score=confidence,
        speaker_count=speaker_count,
    )
