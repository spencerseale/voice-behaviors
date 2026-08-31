"""Hardcoded stand-ins for the proprietary per-call config.

In deployment these come from SIP headers, our config service, and per-tenant
prompt storage. Pinned here so the demo runs with nothing but API keys.
"""

import os
from typing import Any

BRAINTRUST_PROJECT = "voice-behaviors"
AGENT_NAME = "braintrust-otel-agent"

# LiveKit Inference model strings (public; swap freely). inference.STT/LLM/TTS
# route through the LiveKit Inference gateway, so no extra provider plugins or
# keys are required beyond LiveKit Cloud.
STT_MODEL = "deepgram/nova-3"
LLM_MODEL = "openai/gpt-4o-mini"
TTS_MODEL = "deepgram/aura-2"
TTS_VOICE = "thalia"

# Session-level TTS watchdog, applied via AgentSession conn_options.
TTS_SESSION_TIMEOUT = 10.0

# --- Simulated caller (eval harness only) ---------------------------------- #
#
# Which provider the *caller* speaks and listens with. The agent always uses
# LiveKit Inference, because the eval must measure the deployed configuration.
#
# "livekit" (default) shares the gateway with the agent -- simple, one key, and it
# is what production uses. "openai" points the caller at a different vendor, which
# has one real advantage: when both sides share a recognizer they share blind
# spots, so a word the agent mishears the caller mishears identically, hiding the
# failure instead of surfacing it. It also halves gateway load.
#
# "openai" needs a real OPENAI_API_KEY. Note the Braintrust proxy does NOT serve
# the audio endpoints (405 on /audio/speech), so a Braintrust key will not work
# here even though it works for the caller's and judge's chat completions.
CALLER_PROVIDER = os.environ.get("CALLER_PROVIDER", "livekit").lower()

CALLER_MODELS = {
    "livekit": {"stt": STT_MODEL, "tts": TTS_MODEL, "default_voice": TTS_VOICE},
    "openai": {
        "stt": "gpt-4o-mini-transcribe",
        "tts": "gpt-4o-mini-tts",
        "default_voice": "alloy",
    },
}

# Personas name a LiveKit/Deepgram voice; keep them distinct per provider so each
# caller still sounds different when the provider is swapped.
_VOICE_MAP = {
    "openai": {
        "thalia": "nova",
        "asteria": "shimmer",
        "orion": "onyx",
        "arcas": "echo",
        "luna": "coral",
    }
}


def caller_voice(voice: str | None) -> str:
    """Translate a persona's voice name into the active provider's namespace."""
    models = CALLER_MODELS[CALLER_PROVIDER]
    if not voice:
        return models["default_voice"]
    return _VOICE_MAP.get(CALLER_PROVIDER, {}).get(voice, voice)

# Per-call identifiers normally pulled from SIP headers / config. Attached to the
# root span so traces are sliceable by tenant/workflow.
DEMO_WORKFLOW_ID = "wf_example_0001"
DEMO_TENANT_ID = "org_example_0001"

DEMO_SYSTEM_PROMPT = (
    "You are a friendly voice assistant for a demo business. Keep replies short "
    "and conversational. Do not read out symbols, markdown, or code."
)
DEMO_GREETING = "Hi! Thanks for calling. How can I help you today?"

# Stand-in for the per-call data normally prefetched and seeded into chat_ctx so
# the LLM has it on turn one without polluting the (cache-stable) system prompt.
DEMO_CALL_DATA: dict[str, Any] = {
    "caller_id": "+15555550123",
    "business_name": "Example Co.",
    "intent": "general_inquiry",
}
