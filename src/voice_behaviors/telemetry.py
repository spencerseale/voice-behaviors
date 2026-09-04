"""Braintrust telemetry setup via the OTel path.

This is the OTel variant of `braintrust.auto_instrument()`. Instead of braintrust
monkey-patching LiveKit into flat native spans, a `BraintrustSpanProcessor` is
registered on an OTel TracerProvider that LiveKit is handed via
`set_tracer_provider`. LiveKit's OWN OpenTelemetry spans
(agent_session -> agent_turn / user_turn -> llm/tts/stt) are exported to
Braintrust as-is, so conversation turns nest.

The `job_entrypoint` root is created as an OTel span (NOT braintrust.start_span),
because the two live in different context systems -- the OTel root is what
LiveKit's `agent_session` span parents under.
"""

import logging
import os

from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.trace import Tracer

from .config import BRAINTRUST_PROJECT
from .masking import MaskingSpanProcessor, noop_masking_function

logger = logging.getLogger(__name__)

# Tracer used to open the `job_entrypoint` OTel root. Set in setup so it comes from
# the SAME provider that carries the BraintrustSpanProcessor (otherwise the root
# span would not be exported to Braintrust). None when tracing is disabled.
_TRACER: Tracer | None = None


def get_tracer() -> Tracer | None:
    """The app tracer, or None when telemetry setup was skipped."""
    return _TRACER


def setup_braintrust_telemetry() -> bool:
    """Register a BraintrustSpanProcessor and hand the provider to LiveKit.

    Called once per worker subprocess from `prewarm()`, before any AgentSession
    exists. Returns True when tracing was configured.
    """
    global _TRACER  # noqa: PLW0603
    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        logger.info("braintrust telemetry disabled (no BRAINTRUST_API_KEY)")
        return False

    from braintrust.otel import BraintrustSpanProcessor
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    # filter_ai_spans=False (the default) keeps LiveKit's lk.* voice spans --
    # AI-span filtering would drop the session/turn/speaking spans.
    #
    # Masking: wrap the BraintrustSpanProcessor so PII is redacted from span
    # attributes/events BEFORE export. This is the OTel-path analogue of
    # braintrust.set_masking_function(...) (which only applies to native-SDK
    # spans and would be a no-op here). noop_masking_function is a passthrough.
    provider.add_span_processor(
        MaskingSpanProcessor(
            BraintrustSpanProcessor(
                api_key=api_key,
                parent=f"project_name:{BRAINTRUST_PROJECT}",
                filter_ai_spans=False,
            ),
            noop_masking_function,
        )
    )
    # Make LiveKit emit ALL of its OTel spans through this provider.
    set_tracer_provider(provider)
    # Root tracer must come from the same provider so job_entrypoint is exported too.
    _TRACER = provider.get_tracer("voice_behaviors.agent")

    logger.info("braintrust OTel telemetry enabled (project=%s)", BRAINTRUST_PROJECT)
    return True
