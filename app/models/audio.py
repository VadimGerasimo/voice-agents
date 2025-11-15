from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


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


class VoiceToVoiceRequest(BaseModel):
    """Request model for voice-to-voice conversation."""

    system_prompt: Optional[str] = Field(
        None,
        description="System prompt to guide the LLM behavior (e.g., 'You are a helpful assistant')"
    )
    language: Optional[str] = Field(
        None,
        description="Language code for audio transcription (e.g., 'en', 'es')"
    )
    llm_model: str = Field(
        "gpt-4o",
        description="LLM model to use for generating responses (default: gpt-4o)"
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the LLM (0-2, default: 0.7)"
    )
    tts_voice: str = Field(
        "alloy",
        description="Voice for text-to-speech (alloy, echo, fable, onyx, nova, shimmer)"
    )
    tts_model: str = Field(
        "tts-1",
        description="TTS model to use (tts-1 for low latency, tts-1-hd for high quality)"
    )
    output_format: str = Field(
        "mp3",
        description="Audio format for output (mp3, opus, aac, flac)"
    )


class VoiceToVoiceResponse(BaseModel):
    """Response model for voice-to-voice conversation."""

    transcribed_text: str = Field(..., description="The user's spoken input (transcribed)")
    response_text: str = Field(..., description="The LLM's text response")
    audio_format: str = Field(..., description="Format of the generated audio")
    audio_bytes: bytes = Field(..., description="The generated speech audio as bytes")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the processing pipeline")


class VoiceToVoiceFileRequest(BaseModel):
    """Request model for voice-to-voice conversation with file output."""

    system_prompt: Optional[str] = Field(None, description="System prompt for the LLM")
    language: Optional[str] = Field(None, description="Language code for transcription")
    llm_model: str = Field("gpt-4o", description="LLM model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature for the LLM")
    tts_voice: str = Field("alloy", description="Voice for text-to-speech")
    tts_model: str = Field("tts-1", description="TTS model to use")
    output_format: str = Field("mp3", description="Audio format for output")


class VoiceToVoiceFileResponse(BaseModel):
    """Response model for voice-to-voice conversation with file output."""

    transcribed_text: str = Field(..., description="The user's spoken input (transcribed)")
    response_text: str = Field(..., description="The LLM's text response")
    output_audio_path: str = Field(..., description="Path to the saved audio file")
    audio_format: str = Field(..., description="Format of the generated audio")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the processing pipeline")
