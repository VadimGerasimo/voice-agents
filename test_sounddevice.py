#!/usr/bin/env python3
"""
Standalone test script to verify sounddevice is working properly.
This helps diagnose audio playback issues.
"""

import sounddevice as sd
import numpy as np
import sys

def list_devices():
    """List all available audio devices."""
    print("\n" + "="*60)
    print("AVAILABLE AUDIO DEVICES")
    print("="*60)
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"[{i}] {device['name']}")
            print(f"    Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
            print(f"    Sample Rate: {device['default_samplerate']} Hz")
            print()
    except Exception as e:
        print(f"ERROR listing devices: {e}")
        return False
    return True

def get_default_device():
    """Get the default output device."""
    print("\n" + "="*60)
    print("DEFAULT OUTPUT DEVICE")
    print("="*60)
    try:
        default = sd.default.device
        device_info = sd.query_devices(default[1])
        print(f"Device ID: {default[1]}")
        print(f"Device Name: {device_info['name']}")
        print(f"Channels: {device_info['max_output_channels']}")
        print(f"Sample Rate: {device_info['default_samplerate']} Hz")
        return True
    except Exception as e:
        print(f"ERROR getting default device: {e}")
        return False

def test_sine_tone():
    """Test playback with a simple sine tone."""
    print("\n" + "="*60)
    print("SINE TONE TEST")
    print("="*60)
    try:
        sample_rate = 24000
        duration = 2  # seconds
        frequency = 440  # Hz (A note)

        print(f"Generating {duration}s sine tone at {frequency} Hz...")

        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3  # 0.3 amplitude to avoid clipping
        audio_int16 = (audio * 32767).astype(np.int16)

        print(f"Audio shape: {audio_int16.shape}")
        print(f"Audio dtype: {audio_int16.dtype}")
        print(f"Audio min/max: {audio_int16.min()}/{audio_int16.max()}")

        print(f"Playing sine tone using sd.play()...")
        sd.play(audio_int16, sample_rate)
        sd.wait()
        print("✓ Sine tone played successfully!")
        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stream_write():
    """Test playback using OutputStream.write()."""
    print("\n" + "="*60)
    print("STREAM WRITE TEST")
    print("="*60)
    try:
        sample_rate = 24000
        duration = 2  # seconds
        frequency = 880  # Hz (higher note)

        print(f"Generating {duration}s sine tone at {frequency} Hz...")

        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_int16 = (audio * 32767).astype(np.int16)

        print(f"Opening OutputStream...")
        stream = sd.OutputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=2048,
            dtype=np.int16,
            latency='low'
        )
        stream.start()
        print("Stream started!")

        print(f"Writing {len(audio_int16)} samples to stream...")
        stream.write(audio_int16)
        print("Waiting for playback...")
        sd.wait()

        stream.stop()
        stream.close()
        print("✓ Stream write test passed!")
        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pcm16_audio():
    """Test playback with PCM16 audio (like OpenAI output)."""
    print("\n" + "="*60)
    print("PCM16 AUDIO TEST (simulating OpenAI output)")
    print("="*60)
    try:
        import base64

        sample_rate = 24000
        duration = 2
        frequency = 1000

        print(f"Generating PCM16 audio...")
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_int16 = (audio * 32767).astype(np.int16)

        # Simulate what OpenAI sends: base64-encoded PCM16
        audio_bytes = audio_int16.tobytes()
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        print(f"Base64 encoded audio: {len(encoded)} chars")

        # Decode and play (like our handler does)
        decoded = base64.b64decode(encoded)
        audio_restored = np.frombuffer(decoded, dtype=np.int16).copy()
        print(f"Decoded audio: {len(audio_restored)} samples")

        print(f"Playing decoded audio...")
        sd.play(audio_restored, sample_rate)
        sd.wait()
        print("✓ PCM16 audio test passed!")
        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SOUNDDEVICE DIAGNOSTIC TEST")
    print("="*60)

    results = {
        "List Devices": list_devices(),
        "Default Device": get_default_device(),
        "Sine Tone": test_sine_tone(),
        "Stream Write": test_stream_write(),
        "PCM16 Audio": test_pcm16_audio(),
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Sounddevice is working properly!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - There may be audio device issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
