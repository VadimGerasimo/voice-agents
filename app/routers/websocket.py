import logging
import asyncio
import json
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import websockets

from app.config import settings
from app.services.vad_service import get_vad_service, StreamingVAD
from app.services.openai_service import openai_service
from app.services.voice_to_voice_service import voice_to_voice_service
import numpy as np
import io
import wave

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

            # Track if session was configured and check for mismatches
            session_configured = False
            session_update_acknowledged = False
            warnings = []

            # Check if using defaults
            default_instructions = "You are a helpful English-speaking assistant. Answer questions concisely and naturally. Speak in a conversational tone."
            default_voice = "alloy"

            if instructions == default_instructions:
                warnings.append("⚠️ Using DEFAULT system instructions")
                logger.warning("System prompt is using DEFAULT value")
            if voice == default_voice:
                warnings.append("⚠️ Using DEFAULT voice (alloy)")
                logger.warning("Voice is using DEFAULT value")

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
                nonlocal session_configured, session_update_acknowledged, warnings
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
                            elif msg_type == 'session.updated':
                                logger.info(f"OpenAI event: {msg_type}")
                                session_configured = True
                                session_update_acknowledged = True

                                # Send warnings to client if using defaults
                                if warnings:
                                    for warning_msg in warnings:
                                        warning_notification = {
                                            "type": "system.warning",
                                            "message": warning_msg
                                        }
                                        await websocket.send_text(json.dumps(warning_notification))
                                        logger.warning(f"Sent warning to client: {warning_msg}")
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


@router.websocket("/smart-voice")
async def websocket_smart_voice_endpoint(
    websocket: WebSocket,
    device_id: int = Query(None),
    voice: str = Query("alloy"),
    instructions: str = Query("You are a helpful English-speaking assistant. Answer questions concisely and naturally. Speak in a conversational tone."),
):
    """
    Smart Voice WebSocket endpoint with Voice Activity Detection (VAD).

    Provides real-time speech detection and automatic STT→LLM→TTS pipeline.

    Client should send audio chunks at 16kHz sample rate as base64-encoded 16-bit PCM.

    Example message:
    {
        "type": "audio_chunk",
        "audio": "base64_encoded_audio"
    }

    Server sends back:
    {
        "type": "vad_status",
        "is_speaking": true,
        "speech_prob": 0.8,
        "speech_duration_ms": 500
    }

    or

    {
        "type": "transcript",
        "user_text": "What time is it?",
        "assistant_text": "It is 3:00 PM."
    }
    """
    await websocket.accept()

    # Initialize VAD
    logger.info("Initializing VAD service...")
    vad_service = get_vad_service()
    logger.info("VAD service obtained, creating StreamingVAD...")
    streaming_vad = StreamingVAD(
        vad_service,
        min_speech_duration_ms=200,  # Minimum 200ms of speech before triggering
        min_silence_duration_ms=1500,  # Wait 1.5 seconds of silence before ending speech (accounts for pauses between words)
        speech_threshold=0.35  # Lower threshold for better speech detection
    )

    # Audio buffer for VAD processing
    SAMPLE_RATE = 16000
    CHUNK_SIZE = int(SAMPLE_RATE * 30 / 1000)  # 30ms chunks

    try:
        logger.info(f"Smart voice client connected - voice: {voice}, instructions: {instructions[:50]}...")
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": "Connected. Start speaking...",
            "sample_rate": SAMPLE_RATE,
            "chunk_size": CHUNK_SIZE
        }))

        chunk_count = 0
        while True:
            try:
                # Receive audio chunk from client with timeout
                try:
                    logger.debug("Waiting for audio chunk from client...")
                    try:
                        message_text = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        logger.debug(f"Received message: {len(message_text)} bytes")
                    except asyncio.TimeoutError:
                        logger.warning("No audio chunks received for 30 seconds - closing connection")
                        break
                except WebSocketDisconnect:
                    logger.info("Client disconnected during receive")
                    break

                try:
                    message = json.loads(message_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message as JSON: {e}")
                    continue

                if message.get('type') == 'audio_chunk':
                    chunk_count += 1
                    # Decode base64 audio
                    audio_base64 = message.get('audio', '')
                    if not audio_base64:
                        continue

                    try:
                        audio_bytes = base64.b64decode(audio_base64)
                        audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                        chunk_size = len(audio_chunk)

                        if chunk_size != 512:
                            logger.warning(f"[CHUNK {chunk_count}] Non-standard size: {chunk_size} samples (expected 512)")
                        else:
                            logger.debug(f"[CHUNK {chunk_count}] Standard size: 512 samples ✓")

                        logger.info(f"[CHUNK {chunk_count}] Received {chunk_size} samples")

                        # Process with VAD
                        vad_result = streaming_vad.process_chunk(audio_chunk)

                        # Log VAD results for debugging
                        if not vad_result['speech_detected']:
                            logger.debug(
                                f"VAD: silence_duration={vad_result['silence_duration_ms']}ms, "
                                f"speech_duration={vad_result['speech_duration_ms']}ms, "
                                f"should_finalize={vad_result['should_finalize']}, "
                                f"is_speaking={vad_result['is_speaking']}"
                            )

                        # Send VAD status back to client
                        await websocket.send_text(json.dumps({
                            "type": "vad_status",
                            "is_speaking": vad_result['is_speaking'],
                            "speech_detected": vad_result['speech_detected'],
                            "speech_prob": round(vad_result['speech_prob'], 3),
                            "speech_duration_ms": vad_result['speech_duration_ms'],
                            "silence_duration_ms": vad_result['silence_duration_ms'],
                            "should_finalize": vad_result['should_finalize'],
                        }))

                        # If speech ended, process it
                        if vad_result['should_finalize']:
                            logger.info(f"Speech finalization triggered by VAD (silence={vad_result['silence_duration_ms']}ms)")
                            logger.info("Speech finalized - processing with STT→LLM→TTS")

                            # Get accumulated speech buffer
                            speech_audio = streaming_vad.get_speech_buffer()
                            if speech_audio is not None:
                                # Create WAV file from audio
                                wav_buffer = io.BytesIO()
                                with wave.open(wav_buffer, 'wb') as wav_file:
                                    wav_file.setnchannels(1)
                                    wav_file.setsampwidth(2)
                                    wav_file.setframerate(SAMPLE_RATE)
                                    wav_file.writeframes(speech_audio.tobytes())

                                wav_buffer.seek(0)
                                wav_buffer.name = "speech.wav"  # OpenAI API needs a name attribute

                                # Send transcribing status
                                await websocket.send_text(json.dumps({
                                    "type": "status",
                                    "message": "Transcribing..."
                                }))

                                try:
                                    # Transcribe
                                    transcription = openai_service.transcribe_audio(
                                        wav_buffer,
                                        language=None
                                    )
                                    user_text = transcription.get('text', '')
                                    logger.info(f"Transcribed: {user_text}")

                                    # Get LLM response
                                    await websocket.send_text(json.dumps({
                                        "type": "status",
                                        "message": "Getting response..."
                                    }))

                                    llm_response = openai_service.get_chat_completion(
                                        messages=[{"role": "user", "content": user_text}],
                                        system_prompt=instructions,
                                        model="gpt-4o",
                                        temperature=0.7
                                    )
                                    assistant_text = llm_response.get('text', '')
                                    logger.info(f"LLM response: {assistant_text}")

                                    # Generate TTS
                                    await websocket.send_text(json.dumps({
                                        "type": "status",
                                        "message": "Generating speech..."
                                    }))

                                    audio_bytes = openai_service.text_to_speech(
                                        assistant_text,
                                        voice=voice,
                                        model="tts-1",
                                        output_format="mp3"
                                    )

                                    # Send transcript and audio
                                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                                    await websocket.send_text(json.dumps({
                                        "type": "transcript",
                                        "user_text": user_text,
                                        "assistant_text": assistant_text,
                                    }))

                                    await websocket.send_text(json.dumps({
                                        "type": "audio",
                                        "audio": audio_base64,
                                        "format": "mp3"
                                    }))

                                    logger.info("Response sent to client")

                                except Exception as e:
                                    logger.error(f"Error processing speech: {e}")
                                    await websocket.send_text(json.dumps({
                                        "type": "error",
                                        "message": f"Error: {str(e)}"
                                    }))

                    except Exception as e:
                        logger.error(f"Error decoding audio chunk: {e}")

                else:
                    logger.debug(f"Received non-audio message type: {message.get('type')}")

            except Exception as e:
                logger.error(f"Error in smart-voice loop: {e}")

    except WebSocketDisconnect:
        logger.info("Smart voice client disconnected")
    except Exception as e:
        logger.error(f"Smart voice WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except:
            pass
    finally:
        logger.info("Smart voice WebSocket connection closed")
        try:
            await websocket.close()
        except:
            pass
