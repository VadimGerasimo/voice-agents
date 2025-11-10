import logging
import asyncio
import base64
import json
from typing import Optional, AsyncGenerator, Callable
from openai import OpenAI, AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize async OpenAI client for realtime API
async_client = AsyncOpenAI(api_key=settings.openai_api_key)


class RealtimeService:
    """Service for OpenAI Realtime API interactions."""

    def __init__(self):
        self.model = "gpt-4o-realtime-mini"
        self.voice = "alloy"
        self.instructions = None

    async def create_session(
        self,
        model: str = "gpt-4o-realtime-mini",
        voice: str = "alloy",
        instructions: Optional[str] = None,
    ) -> dict:
        """
        Create a new realtime session with OpenAI.

        Args:
            model: The realtime model to use
            voice: The voice for audio output
            instructions: System instructions for the AI

        Returns:
            Session configuration dict
        """
        try:
            self.model = model
            self.voice = voice
            self.instructions = instructions

            logger.info(f"Creating realtime session with model: {model}, voice: {voice}")

            # For realtime API, we return session config
            # The actual session is managed through WebSocket
            return {
                "model": model,
                "voice": voice,
                "instructions": instructions,
            }

        except Exception as e:
            logger.error(f"Error creating realtime session: {str(e)}")
            raise

    async def send_audio_input(
        self,
        websocket_conn,
        audio_data: bytes,
    ) -> None:
        """
        Send audio input to OpenAI realtime API.

        Args:
            websocket_conn: WebSocket connection to OpenAI realtime
            audio_data: Raw audio bytes to send
        """
        try:
            # Encode audio as base64
            encoded_audio = base64.b64encode(audio_data).decode("utf-8")

            # Create input audio delta event for realtime API
            event = {
                "type": "input_audio_buffer.append",
                "audio": encoded_audio,
            }

            # Send to OpenAI realtime API
            await websocket_conn.send(json.dumps(event))

        except Exception as e:
            logger.error(f"Error sending audio input: {str(e)}")
            raise

    async def commit_audio_input(self, websocket_conn) -> None:
        """
        Commit the audio input buffer for processing.

        Args:
            websocket_conn: WebSocket connection to OpenAI realtime
        """
        try:
            event = {"type": "input_audio_buffer.commit"}
            await websocket_conn.send(json.dumps(event))

        except Exception as e:
            logger.error(f"Error committing audio input: {str(e)}")
            raise

    async def create_response(self, websocket_conn) -> None:
        """
        Request OpenAI to create a response (transcribe + generate + synthesize).

        Args:
            websocket_conn: WebSocket connection to OpenAI realtime
        """
        try:
            event = {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": self.instructions,
                },
            }
            await websocket_conn.send(json.dumps(event))

        except Exception as e:
            logger.error(f"Error creating response: {str(e)}")
            raise


# Create a singleton instance
realtime_service = RealtimeService()
