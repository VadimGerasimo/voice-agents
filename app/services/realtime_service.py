import logging
import asyncio
import base64
import json
from typing import Optional, AsyncGenerator, Callable
from openai import OpenAI, AsyncOpenAI
import sounddevice as sd
import numpy as np

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

    async def receive_and_play_audio(
        self,
        websocket_conn,
        sample_rate: int = 24000,
        blocksize: int = 4096,
    ) -> None:
        """
        Receive audio messages from OpenAI and play them using sounddevice.

        Listens for response.audio_delta events, decodes base64 audio chunks,
        and plays them in real-time using sounddevice.

        Args:
            websocket_conn: WebSocket connection to OpenAI realtime API
            sample_rate: Sample rate of the audio (default: 24000 Hz for OpenAI)
            blocksize: Block size for audio playback (default: 4096 samples)

        Example:
            await realtime_service.receive_and_play_audio(openai_ws)
        """
        audio_buffer = bytearray()
        stream = None

        try:
            logger.info(f"Starting audio playback handler (sample_rate={sample_rate}, blocksize={blocksize})")

            # Open audio stream for playback
            stream = sd.OutputStream(
                channels=1,  # Mono audio
                samplerate=sample_rate,
                blocksize=blocksize,
                dtype='int16'  # OpenAI uses PCM16 format
            )
            stream.start()
            logger.info("Audio stream opened successfully")

            async for message_text in websocket_conn:
                try:
                    event = json.loads(message_text)
                    event_type = event.get("type")

                    # Handle audio delta events
                    if event_type == "response.audio_delta":
                        audio_delta = event.get("delta")
                        if audio_delta:
                            try:
                                # Decode base64-encoded PCM16 audio
                                audio_chunk = base64.b64decode(audio_delta)

                                # Convert bytes to int16 array
                                audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)

                                # Play audio immediately using sounddevice
                                sd.play(audio_int16, samplerate=sample_rate, blocking=False)

                                # Also buffer for potential post-processing
                                audio_buffer.extend(audio_chunk)

                                if len(audio_buffer) % (blocksize * 100) == 0:
                                    logger.debug(f"Received and played {len(audio_buffer)} bytes of audio")

                            except Exception as e:
                                logger.error(f"Error decoding audio delta: {str(e)}")

                    # Handle response completion
                    elif event_type == "response.completed":
                        logger.info("Response completed. Waiting for audio playback to finish...")
                        # Wait for any remaining audio to finish playing
                        sd.wait()
                        break

                    # Handle errors from OpenAI
                    elif event_type == "error":
                        error_info = event.get("error", {})
                        logger.error(f"OpenAI error: {error_info}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing audio message: {str(e)}", exc_info=True)

        except Exception as e:
            logger.error(f"Error in audio playback handler: {str(e)}", exc_info=True)

        finally:
            # Cleanup
            if stream:
                try:
                    sd.wait()  # Wait for remaining audio to finish
                    stream.stop()
                    stream.close()
                    logger.info("Audio stream closed")
                except Exception as e:
                    logger.error(f"Error closing audio stream: {str(e)}")


# Create a singleton instance
realtime_service = RealtimeService()
