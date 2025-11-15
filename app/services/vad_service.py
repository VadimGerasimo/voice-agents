"""
Voice Activity Detection (VAD) Service using Silero VAD
Detects speech boundaries with high accuracy for conversational AI
"""

import logging
import numpy as np
import torch
from typing import Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)

# VAD Configuration
SAMPLE_RATE = 16000  # Silero VAD works best at 16kHz
CHUNK_DURATION_MS = 30  # 30ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# VAD Parameters
MIN_SILENCE_MS = 500  # Silence duration to detect end of speech
MIN_SPEECH_MS = 100   # Minimum speech duration to consider
SPEECH_PAD_MS = 300   # Padding before speech onset


class VADService:
    """Voice Activity Detection service using Silero VAD"""

    def __init__(self):
        """Initialize VAD service (model loaded lazily)"""
        self.model = None
        self.device = "cpu"  # Use GPU if available: "cuda"
        self._model_loaded = False
        self._model_loading = False

    def _initialize_model(self):
        """Load Silero VAD model (lazy loading)"""
        if self._model_loaded:
            return

        if self._model_loading:
            # Wait for model to finish loading
            import time
            max_wait = 120  # 2 minutes max
            start = time.time()
            while self._model_loading and (time.time() - start) < max_wait:
                time.sleep(0.1)
            return

        self._model_loading = True
        try:
            logger.info("Loading Silero VAD model...")
            # Load model from torch hub
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False  # Use PyTorch model
            )
            self.model.to(self.device)
            self._model_loaded = True
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            self._model_loading = False
            raise
        finally:
            self._model_loading = False

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        min_speech_duration_ms: int = MIN_SPEECH_MS,
        min_silence_duration_ms: int = MIN_SILENCE_MS,
        speech_pad_ms: int = SPEECH_PAD_MS,
    ) -> list:
        """
        Get speech timestamps from audio buffer.

        Args:
            audio: Audio samples as numpy array (16-bit PCM)
            min_speech_duration_ms: Minimum speech duration to consider
            min_silence_duration_ms: Minimum silence duration to end speech
            speech_pad_ms: Padding before speech onset

        Returns:
            List of dicts with 'start' and 'end' keys (in milliseconds)
        """
        if len(audio) == 0:
            return []

        try:
            # Convert int16 to float32
            audio_float32 = audio.astype(np.float32) / 32768.0

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_float32)

            # Get speech timestamps
            speech_timestamps = self.model.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=SAMPLE_RATE,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=False,  # Return in samples
            )

            # Convert samples to milliseconds
            timestamps_ms = []
            for ts in speech_timestamps:
                start_ms = int((ts['start'] / SAMPLE_RATE) * 1000)
                end_ms = int((ts['end'] / SAMPLE_RATE) * 1000)
                timestamps_ms.append({'start': start_ms, 'end': end_ms})

            return timestamps_ms

        except Exception as e:
            logger.error(f"Error in get_speech_timestamps: {e}")
            return []

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Get speech probability for a single audio chunk.

        Args:
            audio_chunk: Audio chunk as numpy array (should be 512 samples for 16kHz)

        Returns:
            Speech probability (0.0 to 1.0)
        """
        try:
            # Ensure model is loaded
            if not self._model_loaded:
                self._initialize_model()

            # VAD requires exactly 512 samples at 16kHz
            # Pad or trim to correct size
            expected_samples = 512
            if len(audio_chunk) != expected_samples:
                logger.warning(f"Audio chunk size mismatch: got {len(audio_chunk)} samples, expected {expected_samples}")

                if len(audio_chunk) < expected_samples:
                    # Pad with zeros
                    audio_chunk = np.pad(audio_chunk, (0, expected_samples - len(audio_chunk)), mode='constant')
                else:
                    # Trim to expected size
                    audio_chunk = audio_chunk[:expected_samples]

            # Convert int16 to float32
            audio_float32 = audio_chunk.astype(np.float32) / 32768.0

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_float32)

            # Get speech probability
            speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()

            return float(speech_prob)

        except Exception as e:
            logger.error(f"Error in get_speech_probability: {e}")
            return 0.0

    def reset_model_state(self):
        """Reset model state for new audio stream"""
        try:
            self.model.reset_states()
        except Exception as e:
            logger.warning(f"Error resetting model state: {e}")


class StreamingVAD:
    """Streaming VAD for real-time speech detection"""

    def __init__(
        self,
        vad_service: VADService,
        min_speech_duration_ms: int = MIN_SPEECH_MS,
        min_silence_duration_ms: int = MIN_SILENCE_MS,
        speech_threshold: float = 0.5,
    ):
        """
        Initialize streaming VAD.

        Args:
            vad_service: VADService instance
            min_speech_duration_ms: Minimum speech duration in milliseconds
            min_silence_duration_ms: Minimum silence duration before end-of-speech
            speech_threshold: Probability threshold for speech detection (0.0-1.0)
        """
        self.vad = vad_service
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_threshold = speech_threshold

        # State tracking
        self.is_speaking = False
        self.speech_start_ms = 0
        self.silence_duration_ms = 0
        self.speech_buffer = []
        self.chunk_count = 0

        # Pre-speech buffer (last 500ms of audio before speech detection)
        self.pre_speech_buffer = deque(maxlen=16)  # ~500ms at 32ms chunks (500/32 ≈ 16)

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process a single audio chunk and detect speech.

        Args:
            audio_chunk: Audio chunk (typically 30ms at 16kHz = 480 samples)

        Returns:
            Dict with keys:
            - 'speech_detected': bool - whether speech is currently detected
            - 'speech_prob': float - probability that audio contains speech
            - 'is_speaking': bool - whether the user is currently speaking
            - 'speech_duration_ms': int - duration of current speech segment
            - 'silence_duration_ms': int - duration of current silence segment
            - 'should_finalize': bool - whether current speech should be finalized
        """
        # Always append chunk to pre-speech buffer (maintains ~500ms of audio before speech detection)
        self.pre_speech_buffer.append(audio_chunk)

        # Get speech probability for this chunk
        speech_prob = self.vad.get_speech_probability(audio_chunk)

        # Update tracking
        self.chunk_count += 1
        chunk_duration_ms = CHUNK_DURATION_MS

        # Detect speech onset
        if speech_prob > self.speech_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_ms = self.chunk_count * chunk_duration_ms

                # Prepend pre-speech buffer to capture audio before speech detection
                logger.info(f"Speech detected at {self.speech_start_ms}ms (prepending {len(self.pre_speech_buffer)} chunks from buffer)")
                self.speech_buffer.extend(self.pre_speech_buffer)

            self.speech_buffer.append(audio_chunk)
            self.silence_duration_ms = 0

            return {
                'speech_detected': True,
                'speech_prob': speech_prob,
                'is_speaking': True,
                'speech_duration_ms': (self.chunk_count * chunk_duration_ms) - self.speech_start_ms,
                'silence_duration_ms': 0,
                'should_finalize': False,
            }

        else:  # Silence detected
            if self.is_speaking:
                self.speech_buffer.append(audio_chunk)
                self.silence_duration_ms += chunk_duration_ms

                total_speech_duration_ms = (self.chunk_count * chunk_duration_ms) - self.speech_start_ms

                # Check if we've had enough silence to end speech
                should_finalize = (
                    self.silence_duration_ms >= self.min_silence_duration_ms
                    and total_speech_duration_ms >= self.min_speech_duration_ms
                )

                if should_finalize:
                    logger.info(
                        f"End of speech detected at {self.chunk_count * chunk_duration_ms}ms "
                        f"(silence: {self.silence_duration_ms}ms, speech duration: {total_speech_duration_ms}ms)"
                    )

                # Log silence accumulation for debugging
                if self.silence_duration_ms % 300 == 0:  # Log every 300ms
                    logger.debug(
                        f"Accumulating silence: {self.silence_duration_ms}ms / {self.min_silence_duration_ms}ms required. "
                        f"Speech duration: {total_speech_duration_ms}ms"
                    )

                return {
                    'speech_detected': False,
                    'speech_prob': speech_prob,
                    'is_speaking': True,  # Still in speech segment
                    'speech_duration_ms': total_speech_duration_ms,
                    'silence_duration_ms': self.silence_duration_ms,
                    'should_finalize': should_finalize,
                }

            else:
                # Silence while not speaking - just accumulate in pre-speech buffer
                return {
                    'speech_detected': False,
                    'speech_prob': speech_prob,
                    'is_speaking': False,
                    'speech_duration_ms': 0,
                    'silence_duration_ms': 0,
                    'should_finalize': False,
                }

    def get_speech_buffer(self) -> Optional[np.ndarray]:
        """Get accumulated speech buffer and reset"""
        if not self.speech_buffer:
            return None

        # Concatenate all chunks
        audio_data = np.concatenate(self.speech_buffer)

        # Reset state
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_duration_ms = 0

        return audio_data

    def reset(self):
        """Reset VAD state for new audio stream"""
        self.is_speaking = False
        self.speech_start_ms = 0
        self.silence_duration_ms = 0
        self.speech_buffer = []
        self.chunk_count = 0
        self.vad.reset_model_state()


# Singleton instance
_vad_service = None


def get_vad_service() -> VADService:
    """Get or create VAD service instance"""
    global _vad_service
    if _vad_service is None:
        _vad_service = VADService()
    return _vad_service
