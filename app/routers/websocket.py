import logging
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import websockets

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ws",
    tags=["WebSocket"],
)

# OpenAI Realtime API endpoint
OPENAI_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"


@router.websocket("/voice")
async def websocket_voice_endpoint(
    websocket: WebSocket,
):
    """
    WebSocket endpoint for real-time voice conversation with OpenAI.

    Acts as a proxy between the client and OpenAI's Realtime API.

    Usage:
        ws://localhost:8000/api/ws/voice

    The model and voice settings should be sent in the first message as session configuration.
    """
    await websocket.accept()

    # Connect to OpenAI Realtime WebSocket API without model in URL
    uri = OPENAI_REALTIME_WS_URL

    try:
        logger.info(f"Client connected. Connecting to OpenAI: {uri}")

        async with websockets.connect(
            uri,
            extra_headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            logger.info("Connected to OpenAI Realtime API")

            # Send default session configuration to OpenAI
            session_config = {
                "type": "session.update",
                "session": {
                    "instructions": "You are a helpful English-speaking assistant. Answer questions concisely and naturally. Speak in a conversational tone.",
                    "voice": "alloy",
                    "modalities": ["audio", "text"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "temperature": 0.7
                }
            }
            await openai_ws.send(json.dumps(session_config))
            logger.info("Session configuration sent to OpenAI")

            async def client_to_openai():
                """Relay messages from client to OpenAI."""
                chunk_count = 0
                try:
                    async for message_text in websocket.iter_text():
                        try:
                            # Parse to get message type for logging
                            client_msg = json.loads(message_text)
                            msg_type = client_msg.get('type', 'unknown')

                            if msg_type == 'input_audio_buffer.append':
                                chunk_count += 1
                                audio_len = len(client_msg.get('audio', ''))
                                if chunk_count % 5 == 0:
                                    logger.info(f"Relaying audio chunk {chunk_count}: {audio_len} bytes")
                            else:
                                logger.info(f"Relaying message: {msg_type}")

                            # Forward message directly to OpenAI (no translation)
                            await openai_ws.send(message_text)

                            # Special handling for end_session
                            if msg_type == "end_session":
                                logger.info("Sending response.create to OpenAI")
                                response_msg = json.dumps({
                                    "type": "response.create",
                                    "response": {
                                        "modalities": ["text", "audio"],
                                        "instructions": "Please respond naturally and helpfully to what the user just said."
                                    }
                                })
                                await asyncio.sleep(0.1)
                                await openai_ws.send(response_msg)
                                logger.info("response.create sent")

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse client message: {message_text[:100]} - {str(e)}")
                        except Exception as e:
                            logger.error(f"Error relaying message: {str(e)}", exc_info=True)

                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                except Exception as e:
                    logger.error(f"Error in client_to_openai: {str(e)}", exc_info=True)

            async def openai_to_client():
                """Relay messages from OpenAI to client."""
                try:
                    async for message_text in openai_ws:
                        try:
                            # Forward OpenAI events directly to client
                            openai_event = json.loads(message_text)
                            msg_type = openai_event.get('type', 'unknown')

                            # Only log non-streaming messages to reduce spam
                            if not msg_type.startswith('response.audio') and msg_type != 'input_audio_buffer.speech_started':
                                logger.info(f"OpenAI event: {msg_type}")

                            await websocket.send_text(message_text)

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse OpenAI message: {message_text[:100]} - {str(e)}")
                        except Exception as e:
                            logger.error(f"Error relaying OpenAI message: {str(e)}", exc_info=True)

                except Exception as e:
                    logger.error(f"Error in openai_to_client: {str(e)}", exc_info=True)

            # Run both directions concurrently
            await asyncio.gather(client_to_openai(), openai_to_client())

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_text(f'{{"error": "{str(e)}"}}')
        except:
            pass

    finally:
        logger.info("Voice WebSocket connection closed")
        try:
            await websocket.close()
        except:
            pass
