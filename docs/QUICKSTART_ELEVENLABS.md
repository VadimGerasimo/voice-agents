# ElevenLabs WebSocket Quick Start Guide

Get your voice-to-voice agent up and running in 5 minutes!

## Prerequisites

1. **ElevenLabs Account & Agent**
   - Sign up at [ElevenLabs](https://elevenlabs.io)
   - Create a Conversational AI agent
   - Get your agent ID from the dashboard

2. **Backend Server Running**
   ```bash
   python -m uvicorn app.main:app --reload
   ```
   Server should be running at `http://localhost:8000`

3. **Python Dependencies** (for client examples)
   ```bash
   pip install websockets
   # Optional for audio handling:
   pip install pyaudio numpy soundfile
   ```

## Step 1: Get Your Agent ID

1. Log in to [ElevenLabs](https://elevenlabs.io)
2. Navigate to "Agents" in the dashboard
3. Find your agent and copy the Agent ID

## Step 2: Test the Connection

### Option A: Quick Python Test

Create a file `test_elevenlabs.py`:

```python
import asyncio
import json
import websockets

async def test():
    # Replace with your agent ID
    agent_id = "agent_3301k9s8e6e5eyxa43qztrneq3my"

    # Connect to your backend
    uri = f"ws://localhost:8000/api/ws/elevenlabs/agent?agent_id={agent_id}"

    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        print("✓ Connected!")

        # Send a contextual message
        message = {
            "type": "contextual_update",
            "data": "Hello, I'm ready to talk!"
        }
        await ws.send(json.dumps(message))
        print("✓ Message sent!")

        # Listen for responses
        print("Listening for responses (10 seconds)...")
        try:
            async for msg_text in asyncio.wait_for(ws, timeout=10):
                msg = json.loads(msg_text)
                msg_type = msg.get('type')

                if msg_type == 'agent_response':
                    print(f"Agent: {msg['data']}")
                elif msg_type == 'user_transcript':
                    print(f"User: {msg['data']}")
                elif msg_type == 'audio':
                    print(f"Received audio ({len(msg['data'])} bytes)")
                elif msg_type == 'error':
                    print(f"Error: {msg['data']}")
        except asyncio.TimeoutError:
            print("✓ Timeout - connection is working!")

        await ws.close()

asyncio.run(test())
```

Run it:
```bash
python test_elevenlabs.py
```

Expected output:
```
Connecting to ws://localhost:8000/api/ws/elevenlabs/agent?agent_id=YOUR_AGENT_ID...
✓ Connected!
✓ Message sent!
Listening for responses (10 seconds)...
Agent: <response from agent>
✓ Timeout - connection is working!
```

### Option B: Browser WebSocket Test

Open browser console and run:

```javascript
const agentId = 'YOUR_AGENT_ID';
const ws = new WebSocket(`ws://localhost:8000/api/ws/elevenlabs/agent?agent_id=${agentId}`);

ws.onopen = () => {
  console.log('✓ Connected!');
  ws.send(JSON.stringify({
    type: 'contextual_update',
    data: 'Hello!'
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(`${msg.type}:`, msg.data);
};

ws.onerror = (error) => console.error('✗ Error:', error);
ws.onclose = () => console.log('Disconnected');
```

## Step 3: Stream Audio from Microphone

### Python Example

```python
import asyncio
import json
import websockets
import pyaudio
import numpy as np
import base64

async def stream_microphone():
    agent_id = "YOUR_AGENT_ID"
    uri = f"ws://localhost:8000/api/ws/elevenlabs/agent?agent_id={agent_id}"

    # Audio settings
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Microphone is recording...")
    print("Ctrl+C to stop")

    async with websockets.connect(uri) as ws:
        print("✓ Connected to agent")

        async def send_audio():
            try:
                while True:
                    # Read from microphone
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_int16 = np.frombuffer(data, dtype=np.int16)

                    # Convert to base64
                    audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()

                    # Send to agent
                    message = {
                        "type": "audio",
                        "data": audio_b64,
                        "sample_rate": RATE
                    }
                    await ws.send(json.dumps(message))

                    # Small delay
                    await asyncio.sleep(0.05)
            except KeyboardInterrupt:
                print("\nStopping...")

        async def receive_messages():
            try:
                async for msg_text in ws:
                    msg = json.loads(msg_text)

                    if msg['type'] == 'user_transcript':
                        print(f"You: {msg['data']}")
                    elif msg['type'] == 'agent_response':
                        print(f"Agent: {msg['data']}")
                    elif msg['type'] == 'error':
                        print(f"✗ Error: {msg['data']}")
            except KeyboardInterrupt:
                pass

        # Run both tasks
        try:
            await asyncio.gather(send_audio(), receive_messages())
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

asyncio.run(stream_microphone())
```

Run it:
```bash
python stream_audio.py
```

### JavaScript Example (Web)

```javascript
const agentId = 'YOUR_AGENT_ID';
const ws = new WebSocket(`ws://localhost:8000/api/ws/elevenlabs/agent?agent_id=${agentId}`);

ws.onopen = async () => {
  console.log('✓ Connected to agent');

  // Request microphone access
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(2048, 1, 1);

  processor.onaudioprocess = (event) => {
    const float32 = event.inputBuffer.getChannelData(0);

    // Convert to int16
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      int16[i] = float32[i] < 0
        ? float32[i] * 0x8000
        : float32[i] * 0x7FFF;
    }

    // Convert to base64
    const bytes = new Uint8Array(int16.buffer);
    const binary = String.fromCharCode(...bytes);
    const b64 = btoa(binary);

    // Send to agent
    ws.send(JSON.stringify({
      type: 'audio',
      data: b64,
      sample_rate: 16000
    }));
  };

  source.connect(processor);
  processor.connect(audioContext.destination);
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'user_transcript') {
    console.log(`You: ${msg.data}`);
  } else if (msg.type === 'agent_response') {
    console.log(`Agent: ${msg.data}`);
  } else if (msg.type === 'audio') {
    playAudio(msg.data);
  }
};
```

## Step 4: Use the Full Example Client

The repository includes a complete example client with features like:
- Microphone input
- File input
- Proper error handling
- Audio playback

```bash
python examples/elevenlabs_websocket_client.py YOUR_AGENT_ID
```

## Common Tasks

### 1. Send Only Audio (No Local Playback)

Don't include the `device_id` parameter and ignore audio messages in your client.

```python
# Connect without device_id
uri = f"ws://localhost:8000/api/ws/elevenlabs/agent?agent_id={agent_id}"

# Then handle messages without playing audio
async for msg in websocket:
    if msg['type'] == 'user_transcript':
        # Process transcript
        pass
```

### 2. Specify Audio Output Device

List available devices:

```python
import sounddevice as sd
print(sd.query_devices())
```

Use in WebSocket URL:

```python
# Use device 2 for audio output
uri = f"ws://localhost:8000/api/ws/elevenlabs/agent?agent_id={agent_id}&device_id=2"
```

### 3. Send Contextual Information

Update the agent's context during conversation:

```python
await ws.send(json.dumps({
    "type": "contextual_update",
    "data": "User is on mobile device"
}))
```

### 4. Handle Interruptions

Listen for interruption events:

```python
if msg['type'] == 'interruption':
    print("User interrupted the agent")
    # Stop playing current audio
    # Clear any pending messages
```

## Troubleshooting

### "Connection refused"
- Ensure backend is running: `python -m uvicorn app.main:app --reload`
- Check the URL is correct

### "Invalid agent ID"
- Verify agent ID is correct in ElevenLabs dashboard
- Agent must be created and active

### "No audio output"
- Check audio device is connected and not muted
- Verify system audio settings
- Try specifying `device_id=0` explicitly

### "Slow response"
- Check network connection
- Reduce audio chunk size
- Monitor server CPU/memory usage

## Next Steps

1. **Production Deployment**
   - Use proper logging and error handling
   - Add authentication/authorization
   - Deploy to cloud (Heroku, AWS, etc.)

2. **Frontend Integration**
   - Build UI for conversation
   - Add recording visualizations
   - Implement user profiles

3. **Advanced Features**
   - Multiple agents
   - Conversation history
   - Custom voice configurations
   - Analytics and monitoring

## Documentation

For detailed information, see:
- [Full WebSocket Documentation](./ELEVENLABS_WEBSOCKET.md)
- [Architecture Overview](../CLAUDE.md)
- [Example Client Code](../examples/elevenlabs_websocket_client.py)

## Support

- Check logs for errors: Look for messages in server console
- Enable debug mode: Set `DEBUG=true` in `.env`
- Test manually with curl: `wscat -c "ws://localhost:8000/api/ws/elevenlabs/agent?agent_id=YOUR_ID"`

---

**You're all set!** 🎉

Start streaming audio to your agent and enjoy real-time voice conversations!
