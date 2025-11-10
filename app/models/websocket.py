from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from enum import Enum


class MessageType(str, Enum):
    """Types of WebSocket messages."""

    # Client to server
    AUDIO_INPUT = "audio_input"
    START_SESSION = "start_session"
    END_SESSION = "end_session"

    # Server to client
    AUDIO_OUTPUT = "audio_output"
    TRANSCRIPT = "transcript"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    ERROR = "error"


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure."""

    type: MessageType = Field(..., description="Type of message")
    data: Any = Field(..., description="Message data")
    timestamp: Optional[float] = Field(None, description="Message timestamp")


class AudioInputMessage(BaseModel):
    """Audio input from client."""

    audio_data: str = Field(..., description="Base64 encoded audio data")
    encoding: str = Field(default="pcm16", description="Audio encoding format")


class AudioOutputMessage(BaseModel):
    """Audio output to client."""

    audio_data: str = Field(..., description="Base64 encoded audio data")
    encoding: str = Field(default="pcm16", description="Audio encoding format")


class TranscriptMessage(BaseModel):
    """Transcript message from AI."""

    text: str = Field(..., description="Transcribed or generated text")
    role: str = Field(..., description="Role (user or assistant)")
    is_final: bool = Field(default=False, description="Whether this is the final transcript")


class SessionConfig(BaseModel):
    """Configuration for WebSocket session."""

    model: str = Field(default="gpt-4o-realtime", description="OpenAI realtime model (gpt-4o-realtime or gpt-4o-realtime-preview)")
    instructions: Optional[str] = Field(None, description="System instructions for the AI")
    voice: str = Field(default="alloy", description="Voice for audio output (alloy, echo, shimmer, fable, onyx, nova, sage)")
    language: str = Field(default="en", description="Language code")
