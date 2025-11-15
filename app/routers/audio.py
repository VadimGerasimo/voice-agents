import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

from app.models.audio import (
    TranscriptionRequest,
    TranscriptionResponse,
    VoiceToVoiceRequest,
    VoiceToVoiceResponse,
    VoiceToVoiceFileRequest,
    VoiceToVoiceFileResponse,
)
from app.services.openai_service import openai_service
from app.services.voice_to_voice_service import voice_to_voice_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/audio",
    tags=["Audio"],
    responses={404: {"description": "Not found"}},
)


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Query(None, description="Language code (e.g., 'en', 'es')"),
) -> TranscriptionResponse:
    """
    Transcribe an audio file to text using OpenAI Whisper API.

    Supported audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm

    Args:
        file: Audio file to transcribe
        language: Optional language code for transcription

    Returns:
        TranscriptionResponse with transcribed text and metadata
    """
    try:
        # Save uploaded file to temporary location with original file extension
        file_extension = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        try:
            # Call OpenAI service to transcribe
            result = openai_service.transcribe_audio(tmp_file_path, language=language)

            logger.info(f"Transcription successful for file: {file.filename}")

            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                duration=0.0,  # TODO: Calculate actual duration
                model="whisper-1",
            )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to transcribe audio. Please try again."
        )


@router.post("/transcribe-url", response_model=TranscriptionResponse)
async def transcribe_audio_from_file(
    request: TranscriptionRequest,
) -> TranscriptionResponse:
    """
    Transcribe an audio file from a file path using OpenAI Whisper API.

    Args:
        request: TranscriptionRequest with file_path and optional language

    Returns:
        TranscriptionResponse with transcribed text and metadata
    """
    try:
        logger.info(f"Transcribing audio from path: {request.file_path}")

        # Verify file exists
        if not os.path.exists(request.file_path):
            raise ValueError(f"File not found at path: {request.file_path}")

        # Call OpenAI service to transcribe
        result = openai_service.transcribe_audio(
            request.file_path, language=request.language
        )

        logger.info(f"Transcription successful for file: {request.file_path}")

        return TranscriptionResponse(
            text=result["text"],
            language=result["language"],
            duration=0.0,  # TODO: Calculate actual duration
            model="whisper-1",
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to transcribe audio. Please try again."
        )


@router.post("/voice-to-voice")
async def voice_to_voice_conversation(
    file: UploadFile = File(..., description="Audio file for voice-to-voice conversation"),
    system_prompt: Optional[str] = Query(None, description="System prompt for the LLM"),
    language: Optional[str] = Query(None, description="Language code (e.g., 'en', 'es')"),
    llm_model: str = Query("gpt-4o", description="LLM model to use"),
    temperature: float = Query(0.7, ge=0.0, le=2.0, description="Sampling temperature for the LLM"),
    tts_voice: str = Query("alloy", description="Voice for text-to-speech"),
    tts_model: str = Query("tts-1", description="TTS model to use"),
    output_format: str = Query("mp3", description="Audio format for output"),
):
    """
    Voice-to-voice conversation endpoint.

    Pipeline:
    1. Transcribe audio to text (Whisper)
    2. Get LLM response (GPT)
    3. Convert response to speech (TTS)
    4. Return audio stream

    Returns the response audio with content type based on output_format.
    """
    tmp_file_path = None
    try:
        # Save uploaded file to temporary location
        file_extension = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        logger.info(f"Processing voice-to-voice for file: {file.filename}")

        # Process through the pipeline
        result = voice_to_voice_service.process_voice_to_voice(
            audio_file_path=tmp_file_path,
            system_prompt=system_prompt,
            language=language,
            llm_model=llm_model,
            temperature=temperature,
            tts_voice=tts_voice,
            tts_model=tts_model,
            output_format=output_format,
        )

        logger.info(f"Voice-to-voice processing successful")

        # Determine content type based on output format
        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
        }
        content_type = content_type_map.get(output_format, "audio/mpeg")

        # Return the audio as a stream
        return StreamingResponse(
            iter([result["audio_bytes"]]),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=response.{output_format}",
                "X-Transcribed-Text": quote(result["transcribed_text"], safe=''),
                "X-Response-Text": quote(result["response_text"], safe=''),
            },
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in voice-to-voice conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process voice-to-voice conversation. Please try again.",
        )
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")


@router.post("/voice-to-voice-json", response_model=VoiceToVoiceFileResponse)
async def voice_to_voice_conversation_json(
    file: UploadFile = File(..., description="Audio file for voice-to-voice conversation"),
    system_prompt: Optional[str] = Query(None, description="System prompt for the LLM"),
    language: Optional[str] = Query(None, description="Language code (e.g., 'en', 'es')"),
    llm_model: str = Query("gpt-4o", description="LLM model to use"),
    temperature: float = Query(0.7, ge=0.0, le=2.0, description="Sampling temperature for the LLM"),
    tts_voice: str = Query("alloy", description="Voice for text-to-speech"),
    tts_model: str = Query("tts-1", description="TTS model to use"),
    output_format: str = Query("mp3", description="Audio format for output"),
):
    """
    Voice-to-voice conversation endpoint with JSON response.

    Pipeline:
    1. Transcribe audio to text (Whisper)
    2. Get LLM response (GPT)
    3. Convert response to speech (TTS)
    4. Save audio to file
    5. Return JSON with file path and metadata

    Useful for getting the file path and detailed metadata about the processing.
    """
    tmp_file_path = None
    output_file_path = None
    try:
        # Save uploaded file to temporary location
        file_extension = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        logger.info(f"Processing voice-to-voice (JSON) for file: {file.filename}")

        # Create output file path for the response audio
        output_dir = tempfile.gettempdir()
        output_file_path = os.path.join(
            output_dir,
            f"response_{int(__import__('time').time())}.{output_format}",
        )

        # Process through the pipeline
        result = voice_to_voice_service.process_voice_to_voice_to_file(
            audio_file_path=tmp_file_path,
            output_audio_path=output_file_path,
            system_prompt=system_prompt,
            language=language,
            llm_model=llm_model,
            temperature=temperature,
            tts_voice=tts_voice,
            tts_model=tts_model,
            output_format=output_format,
        )

        logger.info(f"Voice-to-voice processing (JSON) successful")

        return VoiceToVoiceFileResponse(
            transcribed_text=result["transcribed_text"],
            response_text=result["response_text"],
            output_audio_path=result["output_audio_path"],
            audio_format=output_format,
            metadata=result["metadata"],
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in voice-to-voice (JSON) conversation: {str(e)}")
        # Clean up output file if it was created
        if output_file_path and os.path.exists(output_file_path):
            try:
                os.remove(output_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete output file: {str(cleanup_error)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process voice-to-voice conversation. Please try again.",
        )
    finally:
        # Clean up temporary input file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")
