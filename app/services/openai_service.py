import logging
from typing import Optional
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)


class OpenAIService:
    """Service for OpenAI API interactions (Whisper STT and GPT)."""

    @staticmethod
    def transcribe_audio(file_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe audio file using OpenAI Whisper API.

        Args:
            file_path: Path to the audio file to transcribe
            language: Optional language code (e.g., 'en', 'es'). If not provided, auto-detect.

        Returns:
            Dictionary containing transcribed text and metadata
        """
        try:
            logger.info(f"Transcribing audio file: {file_path}")

            # Open the audio file
            with open(file_path, "rb") as audio_file:
                # Call Whisper API
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,  # Optional: if None, Whisper will auto-detect
                )

            logger.info(f"Successfully transcribed audio: {transcript.text}")

            return {
                "text": transcript.text,
                "language": language or "auto-detected",
            }

        except FileNotFoundError:
            logger.error(f"Audio file not found: {file_path}")
            raise ValueError(f"Audio file not found at path: {file_path}")
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise


# Create a singleton instance
openai_service = OpenAIService()
