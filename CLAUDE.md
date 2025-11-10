# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice Agent Backend is a production-ready FastAPI application that enables voice-based conversations through real-time streaming. It integrates:
- **OpenAI**: Whisper for speech-to-text (STT) and GPT for conversation logic
- **ElevenLabs**: Text-to-speech (TTS) voice generation
- **Supabase**: PostgreSQL database for conversations, messages, and agent configurations
- **WebSockets**: Real-time bidirectional communication for voice streaming

## Architecture

### Core Architecture

The application follows a layered architecture:

1. **API Layer** (`app/routers/`): FastAPI route handlers that define HTTP/WebSocket endpoints
2. **Service Layer** (`app/services/`): Business logic and external API integrations
3. **Data Layer** (`app/models/`, Supabase): Pydantic models for validation and database schema
4. **Configuration** (`app/config.py`): Centralized environment variable management

### Key Components

- **config.py**: Loads all environment variables using Pydantic BaseSettings
- **main.py**: FastAPI app initialization, middleware setup, and root endpoints
- **routers/**: Endpoint handlers
  - `audio.py`: Audio file uploads and streaming endpoints
  - `conversation.py`: Conversation management (create, update, retrieve)
  - `websocket.py`: WebSocket connections for real-time voice streaming
- **services/**: Integration with external APIs and databases
  - `openai_service.py`: Whisper transcription and GPT conversation logic
  - `elevenlabs_service.py`: TTS voice synthesis
  - `supabase_service.py`: Database operations for conversations and messages
- **models/**: Pydantic data models for request/response validation
- **utils/**: Helper functions and utilities

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application
```bash
# Development server with hot reload
python -m uvicorn app.main:app --reload

# Production server (no reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_openai_service.py

# Run specific test function
pytest tests/test_openai_service.py::test_transcribe_audio

# Run with coverage
pytest --cov=app tests/
```

### Code Quality
```bash
# Format code with Black
black app tests

# Lint with Flake8
flake8 app tests --max-line-length=100

# Type checking with mypy
mypy app
```

## Database Schema

Supabase PostgreSQL database should include tables for:
- **agents**: Agent configurations (personality, voice settings)
- **conversations**: Conversation metadata (user_id, agent_id, created_at, updated_at)
- **messages**: Individual messages (conversation_id, role, content, timestamp)

Each service method that interacts with the database should be in `supabase_service.py`.

## Configuration

All environment variables are defined in `.env.example` and loaded in `app/config.py`:
- **API Keys**: OPENAI_API_KEY, ELEVENLABS_API_KEY, SUPABASE_KEY
- **Database**: SUPABASE_URL
- **Server**: HOST, PORT, DEBUG, LOG_LEVEL
- **App**: APP_ENV

## WebSocket Implementation

WebSocket connections are handled in `app/routers/websocket.py`. Key considerations:
- Use `ConnectionManager` pattern for managing multiple active connections
- Implement proper error handling and connection cleanup
- Stream audio data efficiently for real-time performance
- Integrate with services for STT → LLM → TTS pipeline

## Integration Flow

Typical voice conversation flow:
1. **Client** → WebSocket connection to backend
2. **STT**: Audio stream → OpenAI Whisper (via `openai_service.py`)
3. **LLM**: Transcribed text → GPT conversation logic (via `openai_service.py`)
4. **TTS**: Response text → ElevenLabs voice generation (via `elevenlabs_service.py`)
5. **Streaming**: Generated audio → Client via WebSocket

## Code Style

- Follow PEP 8 guidelines
- Use type hints in function signatures
- Validate input with Pydantic models
- Implement proper error handling with FastAPI exception handlers
- Use descriptive variable and function names
- Add docstrings to all functions and classes

## Common Tasks

### Adding a New Endpoint
1. Create route handler in appropriate `routers/` file
2. Define request/response Pydantic models in `models/`
3. Call relevant service methods
4. Add corresponding tests in `tests/`

### Adding a New Service
1. Create file in `services/` (e.g., `app/services/new_service.py`)
2. Implement class with relevant methods
3. Import and use in routers/
4. Add tests in `tests/test_new_service.py`

### Modifying Database Schema
1. Update Supabase directly or use migrations
2. Update corresponding methods in `supabase_service.py`
3. Update Pydantic models in `models/`
4. Test end-to-end in integration tests

## API Documentation

Once running, interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing Strategy

- Unit tests for service methods (test business logic in isolation)
- Integration tests for API endpoints (test with real/mock services)
- Mock external API calls (OpenAI, ElevenLabs) to avoid costs and ensure speed
- Use pytest fixtures for common test setup

## Performance Considerations

- Stream audio data rather than loading entire files into memory
- Implement connection pooling for Supabase database access
- Cache agent configurations to reduce database queries
- Use async/await throughout for non-blocking I/O operations
