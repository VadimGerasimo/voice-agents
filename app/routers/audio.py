import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import Optional
import os
import tempfile
from pathlib import Path

from app.models.audio import TranscriptionRequest, TranscriptionResponse
from app.services.openai_service import openai_service

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
