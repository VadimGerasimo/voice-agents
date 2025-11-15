import logging
import tempfile
import os
from typing import Optional
from pathlib import Path

from app.services.openai_service import openai_service

logger = logging.getLogger(__name__)


class VoiceToVoiceService:
    """
    Service for voice-to-voice conversations.

    Pipeline:
    1. Transcribe audio to text (Whisper)
    2. Get LLM response (GPT)
    3. Convert response to speech (TTS)
    """

    @staticmethod
    def process_voice_to_voice(
        audio_file_path: str,
        system_prompt: Optional[str] = None,
        language: Optional[str] = None,
        llm_model: str = "gpt-4o",
        temperature: float = 0.7,
        tts_voice: str = "alloy",
        tts_model: str = "tts-1",
        output_format: str = "mp3",
    ) -> dict:
        """
        Process an audio file through the voice-to-voice pipeline.

        Pipeline steps:
        1. Transcribe audio to text
        2. Get LLM response
        3. Convert response to speech audio

        Args:
            audio_file_path: Path to input audio file
            system_prompt: Optional system prompt for LLM behavior
            language: Optional language code for transcription
            llm_model: LLM model to use (default: gpt-4o)
            temperature: Sampling temperature for LLM (default: 0.7)
            tts_voice: Voice to use for TTS (default: alloy)
            tts_model: TTS model to use (default: tts-1)
            output_format: Audio format for output (default: mp3)

        Returns:
            Dictionary containing:
            - transcribed_text: Original user input (transcribed)
            - response_text: LLM response
            - audio_bytes: Generated speech audio
            - audio_format: Format of the audio
            - metadata: Additional metadata about the process
        """
        try:
            logger.info("Starting voice-to-voice processing pipeline")

            # Step 1: Transcribe audio to text
            logger.info(f"Step 1: Transcribing audio from {audio_file_path}")
            transcription_result = openai_service.transcribe_audio(
                audio_file_path,
                language=language
            )
            user_text = transcription_result["text"]
            logger.info(f"Transcribed text: {user_text}")

            # Step 2: Get LLM response
            logger.info("Step 2: Getting LLM response")
            chat_completion_result = openai_service.get_chat_completion(
                messages=[{"role": "user", "content": user_text}],
                system_prompt=system_prompt,
                model=llm_model,
                temperature=temperature,
            )
            response_text = chat_completion_result["text"]
            logger.info(f"LLM response: {response_text}")

            # Step 3: Convert response to speech
            logger.info("Step 3: Converting response to speech")
            audio_bytes = openai_service.text_to_speech(
                text=response_text,
                voice=tts_voice,
                model=tts_model,
                output_format=output_format,
            )
            logger.info(f"Generated audio: {len(audio_bytes)} bytes")

            logger.info("Voice-to-voice processing completed successfully")

            return {
                "transcribed_text": user_text,
                "response_text": response_text,
                "audio_bytes": audio_bytes,
                "audio_format": output_format,
                "metadata": {
                    "transcription": transcription_result,
                    "llm": {
                        "model": chat_completion_result["model"],
                        "finish_reason": chat_completion_result["finish_reason"],
                        "usage": chat_completion_result["usage"],
                    },
                    "tts": {
                        "voice": tts_voice,
                        "model": tts_model,
                        "audio_size_bytes": len(audio_bytes),
                    },
                },
            }

        except Exception as e:
            logger.error(f"Error in voice-to-voice pipeline: {str(e)}")
            raise

    @staticmethod
    def process_voice_to_voice_to_file(
        audio_file_path: str,
        output_audio_path: str,
        system_prompt: Optional[str] = None,
        language: Optional[str] = None,
        llm_model: str = "gpt-4o",
        temperature: float = 0.7,
        tts_voice: str = "alloy",
        tts_model: str = "tts-1",
        output_format: str = "mp3",
    ) -> dict:
        """
        Process an audio file through the voice-to-voice pipeline and save output to file.

        Args:
            audio_file_path: Path to input audio file
            output_audio_path: Path where to save the output audio
            system_prompt: Optional system prompt for LLM behavior
            language: Optional language code for transcription
            llm_model: LLM model to use
            temperature: Sampling temperature for LLM
            tts_voice: Voice to use for TTS
            tts_model: TTS model to use
            output_format: Audio format for output

        Returns:
            Dictionary containing results and file path
        """
        try:
            logger.info("Starting voice-to-voice processing with file output")

            # Step 1: Transcribe audio to text
            logger.info(f"Step 1: Transcribing audio from {audio_file_path}")
            transcription_result = openai_service.transcribe_audio(
                audio_file_path,
                language=language
            )
            user_text = transcription_result["text"]
            logger.info(f"Transcribed text: {user_text}")

            # Step 2: Get LLM response
            logger.info("Step 2: Getting LLM response")
            chat_completion_result = openai_service.get_chat_completion(
                messages=[{"role": "user", "content": user_text}],
                system_prompt=system_prompt,
                model=llm_model,
                temperature=temperature,
            )
            response_text = chat_completion_result["text"]
            logger.info(f"LLM response: {response_text}")

            # Step 3: Convert response to speech and save to file
            logger.info(f"Step 3: Converting response to speech and saving to {output_audio_path}")
            saved_path = openai_service.text_to_speech_to_file(
                text=response_text,
                output_path=output_audio_path,
                voice=tts_voice,
                model=tts_model,
                output_format=output_format,
            )
            logger.info(f"Audio saved to: {saved_path}")

            logger.info("Voice-to-voice processing with file output completed successfully")

            return {
                "transcribed_text": user_text,
                "response_text": response_text,
                "output_audio_path": saved_path,
                "audio_format": output_format,
                "metadata": {
                    "transcription": transcription_result,
                    "llm": {
                        "model": chat_completion_result["model"],
                        "finish_reason": chat_completion_result["finish_reason"],
                        "usage": chat_completion_result["usage"],
                    },
                    "tts": {
                        "voice": tts_voice,
                        "model": tts_model,
                    },
                },
            }

        except Exception as e:
            logger.error(f"Error in voice-to-voice pipeline: {str(e)}")
            raise


# Create a singleton instance
voice_to_voice_service = VoiceToVoiceService()
