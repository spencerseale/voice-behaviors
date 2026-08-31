"""LiveKit voice agent traced into Braintrust over OpenTelemetry.

LiveKit's own OTel spans (agent_session -> agent_turn / user_turn -> llm/tts/stt)
are exported straight to Braintrust, so conversation turns nest in the trace tree.
Span content passes through a masking processor on the way out.
"""

from .agent import Assistant
from .masking import MaskingSpanProcessor, noop_masking_function
from .telemetry import setup_braintrust_telemetry
from .worker import build_session, create_server, handle_session, prewarm

__all__ = [
    "Assistant",
    "MaskingSpanProcessor",
    "build_session",
    "create_server",
    "handle_session",
    "noop_masking_function",
    "prewarm",
    "setup_braintrust_telemetry",
]
