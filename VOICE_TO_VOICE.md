# Voice-to-Voice Functionality

## Overview

The voice-to-voice feature enables end-to-end audio conversations using OpenAI models:

1. **Transcription** (Whisper): Converts user's spoken audio to text
2. **LLM Response** (GPT): Generates intelligent responses based on the transcribed text
3. **Text-to-Speech** (TTS): Converts the response back to natural-sounding audio

## Architecture

### Service Layer

#### `openai_service.py`
Extended with three new methods for complete pipeline support:

- **`transcribe_audio(file_path, language)`**: Converts audio files to text using Whisper
- **`get_chat_completion(messages, system_prompt, model, temperature)`**: Gets LLM responses
- **`text_to_speech(text, voice, model, output_format)`**: Generates speech from text (returns bytes)
- **`text_to_speech_to_file(text, output_path, voice, model, output_format)`**: Generates speech and saves to file

#### `voice_to_voice_service.py` (New)
Orchestrates the complete pipeline:

- **`process_voice_to_voice()`**: Processes audio through the full pipeline and returns audio bytes
- **`process_voice_to_voice_to_file()`**: Processes audio through the pipeline and saves output to file

### API Endpoints

#### 1. Audio Stream Response: `/api/audio/voice-to-voice`
**Method**: `POST`

**Parameters** (Query):
- `file` (Form): Audio file to process (required)
- `system_prompt` (string): Optional system instructions for the LLM
- `language` (string): Language code for transcription (e.g., 'en', 'es')
- `llm_model` (string): LLM model to use (default: 'gpt-4o')
- `temperature` (float): Sampling temperature 0-2 (default: 0.7)
- `tts_voice` (string): Voice for TTS - alloy, echo, fable, onyx, nova, shimmer (default: 'alloy')
- `tts_model` (string): TTS model - tts-1 or tts-1-hd (default: 'tts-1')
- `output_format` (string): Audio format - mp3, opus, aac, flac (default: 'mp3')

**Response**:
- Status: 200 (audio stream)
- Content-Type: Based on output_format (e.g., audio/mpeg)
- Headers:
  - `X-Transcribed-Text`: Original transcribed user input
  - `X-Response-Text`: LLM's text response
  - `Content-Disposition`: Suggests filename

**Example Usage**:
```bash
curl -X POST \
  -F "file=@audio.wav" \
  -H "Content-Type: multipart/form-data" \
  "http://localhost:8000/api/audio/voice-to-voice?system_prompt=You+are+a+helpful+assistant&tts_voice=nova" \
  --output response.mp3
```

**Python Example**:
```python
import requests

with open('audio.wav', 'rb') as f:
    files = {'file': f}
    params = {
        'system_prompt': 'You are a helpful assistant',
        'tts_voice': 'nova',
        'temperature': 0.7
    }
    response = requests.post(
        'http://localhost:8000/api/audio/voice-to-voice',
        files=files,
        params=params
    )

    # Get transcribed and response text from headers
    transcribed = response.headers.get('X-Transcribed-Text')
    response_text = response.headers.get('X-Response-Text')

    # Save audio response
    with open('response.mp3', 'wb') as out:
        out.write(response.content)
```

#### 2. JSON Response: `/api/audio/voice-to-voice-json`
**Method**: `POST`

**Same Parameters as `/voice-to-voice`**

**Response**:
```json
{
  "transcribed_text": "What is the weather like?",
  "response_text": "I don't have access to real-time weather data, but you can check weather.com or ask your local weather service.",
  "output_audio_path": "/tmp/response_1699564800.mp3",
  "audio_format": "mp3",
  "metadata": {
    "transcription": {
      "text": "What is the weather like?",
      "language": "en"
    },
    "llm": {
      "model": "gpt-4o",
      "finish_reason": "stop",
      "usage": {
        "prompt_tokens": 28,
        "completion_tokens": 25,
        "total_tokens": 53
      }
    },
    "tts": {
      "voice": "alloy",
      "model": "tts-1",
      "audio_size_bytes": 45234
    }
  }
}
```

**Python Example**:
```python
import requests

with open('audio.wav', 'rb') as f:
    files = {'file': f}
    params = {
        'system_prompt': 'You are a helpful assistant',
        'tts_voice': 'nova'
    }
    response = requests.post(
        'http://localhost:8000/api/audio/voice-to-voice-json',
        files=files,
        params=params
    )

    data = response.json()
    print(f"User said: {data['transcribed_text']}")
    print(f"Assistant said: {data['response_text']}")
    print(f"Audio saved to: {data['output_audio_path']}")
    print(f"Tokens used: {data['metadata']['llm']['usage']['total_tokens']}")
```

## TTS Voice Options

Available voices for text-to-speech (use `tts_voice` parameter):

- **alloy**: Friendly, neutral
- **echo**: Clear, direct
- **fable**: Warm, narrative
- **onyx**: Deep, confident
- **nova**: Bright, energetic
- **shimmer**: Light, pleasant
- **marin**: (v1 only)
- **cedar**: (v1 only)

## Model Selection

### LLM Models (`llm_model` parameter)
- `gpt-4o`: Latest, most capable (recommended)
- `gpt-4-turbo`: Fast, powerful
- `gpt-3.5-turbo`: Fast, cost-effective
- Other available models as per your OpenAI plan

### TTS Models (`tts_model` parameter)
- `tts-1`: Faster, lower latency, good quality
- `tts-1-hd`: Slower, higher quality audio

### Audio Formats (`output_format` parameter)
- `mp3`: Most compatible (default)
- `opus`: Smaller file size, good for streaming
- `aac`: Good compression
- `flac`: Lossless, largest file size

## Usage Examples

### Basic Voice Conversation
```bash
curl -X POST \
  -F "file=@question.wav" \
  "http://localhost:8000/api/audio/voice-to-voice" \
  --output answer.mp3
```

### With System Prompt
```bash
curl -X POST \
  -F "file=@question.wav" \
  --data-urlencode "system_prompt=You are a pirate. Answer like a pirate." \
  "http://localhost:8000/api/audio/voice-to-voice?tts_voice=onyx" \
  --output answer.mp3
```

### With Custom Settings
```bash
curl -X POST \
  -F "file=@question.wav" \
  "http://localhost:8000/api/audio/voice-to-voice?system_prompt=Be+concise&llm_model=gpt-4o&tts_voice=nova&temperature=0.5" \
  --output answer.mp3
```

### Get Metadata
```bash
curl -X POST \
  -F "file=@question.wav" \
  "http://localhost:8000/api/audio/voice-to-voice-json?system_prompt=Be+helpful" \
  | jq '.'
```

## Advanced Usage

### Streaming Response (Audio)
The `/voice-to-voice` endpoint returns a streaming audio response, making it ideal for:
- Direct audio playback in web browsers
- Streaming responses to clients
- Real-time audio pipelines

### File-based Response (JSON)
The `/voice-to-voice-json` endpoint saves audio to a file and returns metadata, making it ideal for:
- Processing and storing complete conversations
- Getting detailed metrics (tokens used, etc.)
- Accessing the file path for further processing

### Cost Optimization
1. Use `tts-1` instead of `tts-1-hd` for faster responses
2. Use `gpt-3.5-turbo` for simpler queries
3. Use `opus` format for smaller file sizes
4. Set appropriate `temperature` (lower = deterministic, higher = creative)

## Error Handling

All endpoints return meaningful HTTP status codes:

- `200`: Success
- `400`: Invalid request (bad parameters)
- `500`: Server error (API error, file processing issue)

Error responses include a `detail` field with the error message:
```json
{
  "detail": "Failed to process voice-to-voice conversation. Please try again."
}
```

## Implementation Details

### Pipeline Flow

1. **File Upload & Validation**
   - Receives audio file (supports mp3, mp4, mpeg, mpga, m4a, wav, webm)
   - Saves to temporary file
   - Validates language parameter if provided

2. **Transcription**
   - Sends audio to OpenAI Whisper API
   - Returns transcribed text
   - Detects or validates language

3. **LLM Processing**
   - Creates chat message with transcribed text
   - Applies system prompt if provided
   - Uses specified model and temperature
   - Returns response text and usage metrics

4. **Text-to-Speech**
   - Converts response text to speech
   - Uses specified voice and model
   - Generates audio in requested format

5. **Response**
   - `/voice-to-voice`: Streams audio directly
   - `/voice-to-voice-json`: Saves file and returns metadata
   - Both endpoints include transcribed and response text

### Cleanup
- Temporary input files are automatically deleted
- JSON endpoint deletes output file if processing fails
- File permissions are automatically handled

## Performance Characteristics

### Typical Latency (depends on API load)
- **Transcription**: 2-5 seconds for 30-second audio
- **LLM Response**: 1-3 seconds for average queries
- **TTS Generation**: 2-4 seconds for 30-second audio
- **Total**: 5-12 seconds end-to-end

### File Sizes
- **Audio Input**: Depends on codec (typically 100KB-1MB for speech)
- **MP3 Output**: ~50KB per 10 seconds of speech
- **OPUS Output**: ~20KB per 10 seconds of speech
- **FLAC Output**: ~200KB per 10 seconds of speech

## Configuration

All services use the OpenAI API key from `.env`:
```
OPENAI_API_KEY=sk-...
```

Ensure your API key has access to:
- Whisper API (audio/transcriptions)
- Chat Completions API (chat/completions)
- Text-to-Speech API (audio/speech)

## Troubleshooting

### "Audio file not found"
- Ensure the audio file is properly uploaded
- Check file format is supported

### "Failed to transcribe audio"
- Verify audio quality
- Check OPENAI_API_KEY is valid
- Ensure you have Whisper API access

### "Error getting chat completion"
- Check system_prompt is valid
- Verify llm_model exists and is accessible
- Check API quota limits

### "Error converting text to speech"
- Ensure tts_voice is valid
- Check tts_model is correct (tts-1 or tts-1-hd)
- Verify text length (max is usually reasonable)

## Testing

You can test using the Swagger UI:
1. Navigate to `http://localhost:8000/docs`
2. Expand the Audio section
3. Try the `/api/audio/voice-to-voice` endpoint
4. Upload an audio file and adjust parameters
5. Listen to the response audio

## Future Enhancements

- WebSocket support for real-time streaming
- Conversation history and context management
- Audio file caching for repeated requests
- Language detection and auto-selection
- Custom voice cloning
- Audio normalization and enhancement
