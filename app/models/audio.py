from pydantic import BaseModel, Field
from typing import Optional


class TranscriptionRequest(BaseModel):
    """Request model for speech-to-text transcription."""

    file_path: str = Field(..., description="Path to the audio file to transcribe")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'es'). If not provided, Whisper will auto-detect.")


class TranscriptionResponse(BaseModel):
    """Response model for speech-to-text transcription."""

    text: str = Field(..., description="The transcribed text from the audio")
    language: str = Field(..., description="The detected or specified language")
    duration: float = Field(..., description="Duration of the audio file in seconds")
    model: str = Field(default="whisper-1", description="The model used for transcription")
