"""Audio-only empathy scorer for simulated voice calls."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from braintrust import Score, wrap_openai

AUDIO_EMPATHY_MODEL = os.environ.get(
    "AUDIO_EMPATHY_MODEL", "gpt-audio-1.5"
)
AUDIO_EMPATHY_SCORER_VERSION = 1

AudioEmpathyLabel = Literal[
    "empathetic", "adequate", "low_empathy", "dismissive", "audio_unusable"
]

AUDIO_EMPATHY_SCORES: dict[AudioEmpathyLabel, float] = {
    "empathetic": 1.0,
    "adequate": 0.6,
    "low_empathy": 0.25,
    "dismissive": 0.0,
    "audio_unusable": 0.0,
}

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = """You are an evaluator judging empathy in a voice agent call.

You must base the judgment on the supplied audio file, not on a transcript. The WAV
is stereo: the caller is in the left channel and the agent under test is in the
right channel. Listen to the entire file before deciding. Use the caller's speech
only as context; score the agent's spoken wording and delivery.
"""

_USER_PROMPT = """Listen to the full WAV and classify the agent's empathy.

Criterion: Did the agent sound empathic toward the caller in both wording and
delivery?

Classes:
- empathetic: The agent consistently sounds warm, patient, and attentive; it
  acknowledges caller uncertainty, frustration, or needs when those appear.
- adequate: The agent is polite and serviceable, with at least neutral patience;
  it may be transactional but is not cold or dismissive.
- low_empathy: The agent is mostly transactional, rushed, or misses clear chances
  to acknowledge the caller's situation, while not being overtly rude.
- dismissive: The agent sounds impatient, rude, talks over the caller, ignores
  distress or confusion, or otherwise makes the caller feel brushed off.
- audio_unusable: The agent side cannot be heard well enough to judge.

Return JSON only:
{"label":"empathetic|adequate|low_empathy|dismissive|audio_unusable","evidence":"one sentence grounded in the audio"}
"""


@dataclass(frozen=True)
class AudioEmpathyJudgment:
    label: AudioEmpathyLabel
    evidence: str


def _audio_empathy_base_url() -> str:
    return os.environ.get("AUDIO_EMPATHY_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL", "https://gateway.braintrust.dev"
    )


def _audio_empathy_api_key(base_url: str) -> str | None:
    explicit = os.environ.get("AUDIO_EMPATHY_API_KEY")
    if explicit:
        return explicit
    if "gateway.braintrust.dev" in base_url:
        return os.environ.get("BRAINTRUST_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("BRAINTRUST_API_KEY")


@lru_cache(maxsize=1)
def _client() -> Any:
    from openai import AsyncOpenAI

    base_url = _audio_empathy_base_url()
    return wrap_openai(
        AsyncOpenAI(
            base_url=base_url,
            api_key=_audio_empathy_api_key(base_url),
            timeout=90.0,
            max_retries=3,
        )
    )


def _call_audio_bytes(metadata: dict[str, Any] | None) -> bytes:
    if not metadata or "call_audio" not in metadata:
        raise ValueError("audio_empathy requires metadata['call_audio']")

    attachment = metadata["call_audio"]
    if isinstance(attachment, bytes):
        data = attachment
    elif isinstance(attachment, bytearray):
        data = bytes(attachment)
    else:
        data = getattr(attachment, "data", None)

    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("metadata['call_audio'] must expose WAV bytes via .data")
    data = bytes(data)
    if not data:
        raise ValueError("metadata['call_audio'] is empty")
    return data


def _message_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(getattr(part, "text", "")))
        return "".join(text_parts)
    return ""


def _parse_audio_empathy_judgment(raw: str) -> AudioEmpathyJudgment | None:
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    label = str(parsed.get("label", "")).lower()
    if label not in AUDIO_EMPATHY_SCORES:
        return None
    evidence = parsed.get("evidence")
    return AudioEmpathyJudgment(
        label=label,  # type: ignore[arg-type]
        evidence=evidence if isinstance(evidence, str) else "",
    )


async def judge_audio_empathy(wav: bytes) -> AudioEmpathyJudgment:
    wav_b64 = base64.b64encode(wav).decode("ascii")
    response = await _client().chat.completions.create(
        model=AUDIO_EMPATHY_MODEL,
        temperature=0,
        max_completion_tokens=240,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _USER_PROMPT},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": wav_b64, "format": "wav"},
                    },
                ],
            },
        ],
    )
    raw = _message_text(response)
    judgment = _parse_audio_empathy_judgment(raw)
    if judgment is None:
        raise ValueError(
            "audio_empathy judge did not return usable JSON; "
            f"last reply: {raw[:200]!r}"
        )
    return judgment


async def score_audio_empathy(input, metadata=None) -> Score:  # noqa: A002
    # Intentionally do not accept `output`; empathy must be judged from audio bytes.
    wav = _call_audio_bytes(metadata)
    judgment = await judge_audio_empathy(wav)
    return Score(
        name="audio_empathy",
        score=AUDIO_EMPATHY_SCORES[judgment.label],
        metadata={
            "version": AUDIO_EMPATHY_SCORER_VERSION,
            "label": judgment.label,
            "evidence": judgment.evidence,
            "judge_model": AUDIO_EMPATHY_MODEL,
            "input_scope": "root metadata call_audio WAV",
            "audio_format": "wav",
            "audio_bytes": len(wav),
            "transcript_used": False,
        },
    )


score_audio_empathy.__name__ = "audio_empathy"
