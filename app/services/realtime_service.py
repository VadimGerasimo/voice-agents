import logging

logger = logging.getLogger(__name__)


class RealtimeService:
    """
    Service for OpenAI Realtime API interactions.

    Note: Session configuration is handled directly in websocket.py
    Audio playback logic is handled directly in websocket.py
    This class is kept as a placeholder for future service extensions.
    """

    def __init__(self):
        """Initialize RealtimeService."""
        pass


# Create a singleton instance
realtime_service = RealtimeService()
