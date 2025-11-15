import logging
from typing import Optional
import io
from pathlib import Path
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)


class OpenAIService:
    """Service for OpenAI API interactions (Whisper STT, GPT, and TTS)."""

    @staticmethod
    def transcribe_audio(file_path_or_obj, language: Optional[str] = None) -> dict:
        """
        Transcribe audio file using OpenAI Whisper API.

        Args:
            file_path_or_obj: Path to the audio file or a file-like object (BytesIO)
            language: Optional language code (e.g., 'en', 'es'). If not provided, auto-detect.

        Returns:
            Dictionary containing transcribed text and metadata
        """
        try:
            # Handle both file paths and file-like objects
            if isinstance(file_path_or_obj, str):
                logger.info(f"Transcribing audio file: {file_path_or_obj}")
                with open(file_path_or_obj, "rb") as audio_file:
                    # Call Whisper API
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language,  # Optional: if None, Whisper will auto-detect
                    )
            else:
                # Handle file-like object (BytesIO, etc.)
                logger.info("Transcribing audio from file-like object")
                # Call Whisper API with file-like object
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file_path_or_obj,
                    language=language,  # Optional: if None, Whisper will auto-detect
                )

            logger.info(f"Successfully transcribed audio: {transcript.text}")

            return {
                "text": transcript.text,
                "language": language or "auto-detected",
            }

        except FileNotFoundError as e:
            logger.error(f"Audio file not found: {e}")
            raise ValueError(f"Audio file not found: {e}")
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise

    @staticmethod
    def get_chat_completion(
        messages: list,
        system_prompt: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ) -> dict:
        """
        Get chat completion from OpenAI GPT API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            system_prompt: Optional system prompt to set assistant behavior
            model: Model to use (default: gpt-4o)
            temperature: Sampling temperature (0-2, default: 0.7)

        Returns:
            Dictionary containing the response text and metadata
        """
        try:
            logger.info(f"Getting chat completion from {model}")

            # Prepare messages with optional system prompt
            if system_prompt:
                chat_messages = [
                    {"role": "system", "content": system_prompt},
                    *messages,
                ]
            else:
                chat_messages = messages

            # Call Chat Completions API
            response = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=temperature,
            )

            content = response.choices[0].message.content
            logger.info(f"Successfully received chat completion")

            return {
                "text": content,
                "model": model,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        except Exception as e:
            logger.error(f"Error getting chat completion: {str(e)}")
            raise

    @staticmethod
    def text_to_speech(
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        output_format: str = "mp3",
    ) -> bytes:
        """
        Convert text to speech using OpenAI TTS API.

        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer) - default: alloy
            model: Model to use (tts-1 for low latency, tts-1-hd for high quality) - default: tts-1
            output_format: Audio format (mp3, opus, aac, flac) - default: mp3

        Returns:
            Binary audio data
        """
        try:
            logger.info(f"Converting text to speech with voice: {voice}")

            # Call Text-to-Speech API
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=output_format,
            )

            # Get the audio content as bytes
            audio_data = response.content

            logger.info(f"Successfully generated speech audio ({len(audio_data)} bytes)")

            return audio_data

        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            raise

    @staticmethod
    def text_to_speech_to_file(
        text: str,
        output_path: str,
        voice: str = "alloy",
        model: str = "tts-1",
        output_format: str = "mp3",
    ) -> str:
        """
        Convert text to speech and save to file using OpenAI TTS API.

        Args:
            text: Text to convert to speech
            output_path: Path where to save the audio file
            voice: Voice to use - default: alloy
            model: Model to use - default: tts-1
            output_format: Audio format - default: mp3

        Returns:
            Path to the saved audio file
        """
        try:
            logger.info(f"Converting text to speech and saving to: {output_path}")

            # Call Text-to-Speech API with streaming
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                response_format=output_format,
            ) as response:
                response.stream_to_file(output_path)

            logger.info(f"Successfully saved speech audio to {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error saving speech to file: {str(e)}")
            raise


# Create a singleton instance
openai_service = OpenAIService()
