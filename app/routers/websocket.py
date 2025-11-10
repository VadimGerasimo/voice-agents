import logging
import asyncio
import json
import base64
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
    device_id: int = Query(None),
    voice: str = Query("alloy"),
    instructions: str = Query("You are a helpful English-speaking assistant. Answer questions concisely and naturally. Speak in a conversational tone."),
):
    """
    WebSocket endpoint for real-time voice conversation with OpenAI.

    Acts as a proxy between the client and OpenAI's Realtime API.

    Usage:
        ws://localhost:8000/api/ws/voice
        ws://localhost:8000/api/ws/voice?device_id=5&voice=echo&instructions=Be+a+pirate

    Query Parameters:
        device_id: Optional audio device ID to use for playback (default: system default)
        voice: OpenAI voice to use for responses
               Valid voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar
        instructions: System instructions for the AI assistant

    Example:
        voice=coral&instructions=You%20are%20a%20helpful%20assistant
        device_id=5&voice=shimmer&instructions=Be%20a%20pirate
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

            # Send dynamic session configuration to OpenAI
            session_config = {
                "type": "session.update",
                "session": {
                    "instructions": instructions,
                    "voice": voice,
                    "modalities": ["audio", "text"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "temperature": 0.7
                }
            }
            await openai_ws.send(json.dumps(session_config))
            logger.info(f"Session configuration sent to OpenAI - voice: {voice}, instructions: {instructions[:50]}...")

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
                                logger.info(f"[BROWSER] Audio chunk {chunk_count}: {audio_len} bytes")
                            elif msg_type == 'input_audio_buffer.commit':
                                logger.info(f"[BROWSER] Audio commit received ({chunk_count} chunks sent)")
                            else:
                                logger.info(f"[BROWSER] Message: {msg_type}")

                            # Forward message directly to OpenAI (no translation)
                            await openai_ws.send(message_text)
                            logger.debug(f"[BROWSER] Forwarded to OpenAI: {msg_type}")

                        except json.JSONDecodeError as e:
                            logger.error(f"[BROWSER] Failed to parse client message: {message_text[:100]} - {str(e)}")
                        except Exception as e:
                            logger.error(f"[BROWSER] Error relaying message: {str(e)}", exc_info=True)

                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                except Exception as e:
                    logger.error(f"Error in client_to_openai: {str(e)}", exc_info=True)

            # Queue for passing messages from OpenAI to audio playback handler
            audio_queue = asyncio.Queue()

            async def openai_to_client_and_audio():
                """Relay messages from OpenAI to client and queue for audio playback."""
                message_count = 0
                try:
                    async for message in openai_ws:
                        message_count += 1

                        # Check if OpenAI sent binary PCM data
                        if isinstance(message, bytes):
                            logger.info(f"[AUDIO] Message #{message_count}: Received binary audio data: {len(message)} bytes")

                            # Base64-encode the raw PCM audio
                            encoded = base64.b64encode(message).decode("utf-8")

                            # Wrap it as a JSON event the frontend understands
                            audio_event = {
                                "type": "response.audio_delta",
                                "delta": encoded
                            }

                            # Send to browser
                            await websocket.send_text(json.dumps(audio_event))

                            # Also put in queue for local playback
                            await audio_queue.put(json.dumps(audio_event))
                            continue

                        # Otherwise handle as text (JSON) event
                        try:
                            openai_event = json.loads(message)
                            msg_type = openai_event.get('type', 'unknown')

                            # Handle different message types with correct naming (dots, not underscores)
                            if msg_type == 'response.audio.delta':
                                logger.info(f"[AUDIO] Message #{message_count}: AUDIO DELTA RECEIVED!")
                            elif not msg_type.startswith('response.audio') and msg_type != 'input_audio_buffer.speech_started':
                                logger.info(f"OpenAI event: {msg_type}")

                            await websocket.send_text(message)

                            # Queue audio-related events for sounddevice playback
                            # Note: OpenAI uses dots in type names (response.audio.delta, not response.audio_delta)
                            if msg_type in ['response.audio.delta', 'response.done', 'error']:
                                logger.info(f"[AUDIO] Queuing message type: {msg_type}")
                                await audio_queue.put(message)

                        except json.JSONDecodeError as e:
                            logger.warning(f"Received non-JSON text frame from OpenAI: {message[:100]}")
                        except Exception as e:
                            logger.error(f"Error relaying OpenAI message: {str(e)}", exc_info=True)

                except Exception as e:
                    logger.error(f"Error in openai_to_client_and_audio: {str(e)}", exc_info=True)

            async def receive_and_play_audio_from_queue():
                """Receive audio events from queue and play them using sounddevice."""
                import base64
                import numpy as np
                import sounddevice as sd

                stream = None
                audio_buffer = np.array([], dtype=np.int16)
                chunks_received = 0
                total_samples = 0

                try:
                    logger.info("Starting audio playback handler (from queue)")

                    # List available audio devices
                    try:
                        devices = sd.query_devices()
                        logger.info(f"Available audio devices: {devices}")
                        default_device = sd.default.device[1] if sd.default.device else None
                        logger.info(f"Default output device: {default_device}")
                    except Exception as e:
                        logger.warning(f"Could not query audio devices: {e}")
                        default_device = None

                    # Determine which device to use
                    output_device = device_id if device_id is not None else None
                    if output_device is not None:
                        try:
                            device_info = sd.query_devices(output_device)
                            logger.info(f"[AUDIO] Using specified device #{output_device}: {device_info['name']}")
                        except Exception as e:
                            logger.warning(f"[AUDIO] Invalid device ID {output_device}: {e}, using default")
                            output_device = None
                    else:
                        if default_device is not None:
                            try:
                                device_info = sd.query_devices(default_device)
                                logger.info(f"[AUDIO] Using default output device #{default_device}: {device_info['name']}")
                            except Exception as e:
                                logger.warning(f"Could not get default device info: {e}")
                        else:
                            logger.info("[AUDIO] Using system default output device")

                    # Open audio stream for playback with latency control
                    try:
                        stream = sd.OutputStream(
                            device=output_device,  # Use specified or default device
                            channels=1,  # Mono audio
                            samplerate=24000,
                            blocksize=2048,  # Smaller blocksize for lower latency
                            dtype=np.int16,
                            latency='low'
                        )
                        stream.start()
                        logger.info(f"[AUDIO] Audio stream opened successfully (latency='low', device={output_device})")
                    except Exception as e:
                        logger.error(f"[AUDIO] Failed to open audio stream: {e}", exc_info=True)
                        raise

                    while True:
                        try:
                            # Get message from queue with timeout to prevent hanging
                            message_text = await asyncio.wait_for(audio_queue.get(), timeout=30.0)

                            try:
                                event = json.loads(message_text)
                                event_type = event.get("type")

                                # Handle audio delta events (note: OpenAI uses dots, not underscores)
                                if event_type == "response.audio.delta":
                                    audio_delta = event.get("delta")
                                    if audio_delta:
                                        try:
                                            # Decode base64-encoded PCM16 audio
                                            audio_chunk = base64.b64decode(audio_delta)
                                            chunks_received += 1
                                            logger.info(f"[AUDIO] Chunk {chunks_received}: Received {len(audio_chunk)} bytes")

                                            # Convert bytes to int16 array
                                            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16).copy()
                                            total_samples += len(audio_int16)

                                            # Buffer the audio data
                                            audio_buffer = np.concatenate([audio_buffer, audio_int16])

                                            # Verify stream is active
                                            if stream is None:
                                                logger.error("[AUDIO] ERROR: Stream is None!")
                                            elif not stream.active:
                                                logger.error("[AUDIO] ERROR: Stream is not active!")
                                            else:
                                                # Write buffered audio to stream when we have enough data
                                                # Write in chunks to the stream instead of using sd.play
                                                try:
                                                    stream.write(audio_int16)
                                                    logger.info(f"[AUDIO] Chunk {chunks_received}: Wrote {len(audio_int16)} samples to stream (total: {total_samples})")
                                                except Exception as write_error:
                                                    logger.error(f"[AUDIO] ERROR writing to stream: {write_error}", exc_info=True)

                                        except Exception as e:
                                            logger.error(f"[AUDIO] Error decoding/writing audio delta: {str(e)}", exc_info=True)

                                # Handle response completion
                                elif event_type == "response.done":
                                    logger.info(f"[AUDIO] Response done! Total chunks: {chunks_received}, Total samples: {total_samples}")
                                    # Flush remaining audio
                                    if len(audio_buffer) > 0:
                                        logger.info(f"[AUDIO] Flushing remaining {len(audio_buffer)} audio samples")

                                    if stream and stream.active:
                                        try:
                                            logger.info("[AUDIO] Waiting for audio playback to finish...")
                                            # Use a timeout to prevent blocking indefinitely
                                            # Run sd.wait() in a thread to avoid blocking the event loop
                                            loop = asyncio.get_event_loop()
                                            await asyncio.wait_for(
                                                loop.run_in_executor(None, sd.wait),
                                                timeout=5.0  # 5 second timeout
                                            )
                                            logger.info("[AUDIO] Audio playback finished")
                                        except asyncio.TimeoutError:
                                            logger.warning("[AUDIO] Audio playback timeout - continuing anyway")
                                        except Exception as e:
                                            logger.error(f"[AUDIO] Error waiting for audio: {e}")
                                    else:
                                        logger.warning("[AUDIO] Stream is not active, cannot wait")

                                    # Reset for next response
                                    audio_buffer = np.array([], dtype=np.int16)
                                    chunks_received = 0
                                    total_samples = 0

                                # Handle errors from OpenAI
                                elif event_type == "error":
                                    error_info = event.get("error", {})
                                    logger.error(f"OpenAI error: {error_info}")

                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse queued message: {str(e)}")
                            except Exception as e:
                                logger.error(f"Error processing audio from queue: {str(e)}", exc_info=True)

                        except asyncio.TimeoutError:
                            logger.debug("Audio queue timeout, continuing...")
                            continue
                        except asyncio.CancelledError:
                            logger.info("Audio playback handler cancelled")
                            break
                        except Exception as e:
                            logger.error(f"Error in audio playback loop: {str(e)}", exc_info=True)
                            break

                except Exception as e:
                    logger.error(f"Error in audio playback handler: {str(e)}", exc_info=True)

                finally:
                    # Cleanup audio stream
                    if stream:
                        try:
                            # Close the stream gracefully with timeout
                            stream.stop()
                            stream.close()
                            logger.info("Audio stream closed")
                        except Exception as e:
                            logger.error(f"Error closing audio stream: {str(e)}")

            # Run both directions concurrently
            await asyncio.gather(client_to_openai(), openai_to_client_and_audio(), receive_and_play_audio_from_queue())

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
