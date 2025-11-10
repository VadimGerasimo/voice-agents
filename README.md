# Voice Agent Backend

A production-ready backend for voice agent with real-time streaming, speech-to-text transcription, LLM-powered conversations, and text-to-speech voice generation.

## Tech Stack

- **Python 3.11+**
- **FastAPI** - Web framework
- **OpenAI** - Whisper (STT) and GPT (conversations)
- **ElevenLabs** - Text-to-speech voice generation
- **Supabase** - PostgreSQL database
- **WebSockets** - Real-time streaming

## Features

- Audio file uploads and streaming
- Speech-to-text transcription using OpenAI Whisper
- LLM-powered conversation logic with different agent personalities
- Text-to-speech voice generation using ElevenLabs
- Real-time WebSocket connections for voice streaming
- Database storage for conversations, messages, and agent configs

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- API keys for:
  - OpenAI (for Whisper and GPT)
  - ElevenLabs (for TTS)
  - Supabase (PostgreSQL database)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd voice-agent-backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run the application**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`

### Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
voice-agent-backend/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Environment variables & settings
│   ├── models/              # Pydantic models
│   ├── routers/             # API endpoints
│   │   ├── audio.py         # Audio upload and streaming
│   │   ├── conversation.py  # Conversation endpoints
│   │   └── websocket.py     # WebSocket connections
│   ├── services/            # Business logic
│   │   ├── openai_service.py      # OpenAI integration
│   │   ├── elevenlabs_service.py  # ElevenLabs TTS
│   │   └── supabase_service.py    # Database operations
│   └── utils/               # Helper functions
├── tests/                   # Test files
├── requirements.txt         # Python dependencies
├── .env.example             # Example environment variables
├── .gitignore
└── README.md
```

## Development

### Running Tests

```bash
pytest
```

### Running a Single Test

```bash
pytest tests/test_filename.py::test_function_name
```

### Linting and Formatting

```bash
# Install development dependencies
pip install black flake8

# Format code
black app tests

# Check linting
flake8 app tests
```

## Environment Variables

See `.env.example` for all available configuration options.

## API Endpoints

### Health Check

- `GET /health` - Check if service is running
- `GET /` - API information

More endpoints will be added as features are implemented.

## Next Steps

1. Create Pydantic models in `app/models/`
2. Implement service integrations:
   - `app/services/openai_service.py` - Whisper and GPT integration
   - `app/services/elevenlabs_service.py` - TTS integration
   - `app/services/supabase_service.py` - Database operations
3. Create API routers in `app/routers/`:
   - `app/routers/audio.py` - Audio upload/streaming
   - `app/routers/conversation.py` - Conversation management
   - `app/routers/websocket.py` - WebSocket support
4. Add comprehensive tests in `tests/`

## License

MIT
