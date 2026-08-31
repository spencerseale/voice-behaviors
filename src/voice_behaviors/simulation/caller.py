"""The simulated caller: a persona that decides what to say next, and says it.

The caller drives the conversation from the dataset's opening phrase plus a persona
description. It only ever sees what it *heard* -- the transcription of the agent's
audio -- so a reply the agent mangled in synthesis is a reply the caller has to cope
with, exactly as a person would.

The caller's model calls go through the Braintrust proxy and are traced as plain
`caller_generation` spans -- not chat spans. A chat span would record the caller's
next line as an `assistant` message, which makes the trace's thread view show the
agent saying what the caller is about to say.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from braintrust import SpanTypeAttribute, start_span

CALLER_MODEL = os.environ.get("CALLER_MODEL", "claude-sonnet-4-6")
PROXY_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://gateway.braintrust.dev")

# The caller says this (alone) when it is finished; the runner then hangs up.
HANGUP_TOKEN = "[HANGUP]"
NO_COMPLETE_AGENT_REPLY = (
    "(you did not hear a complete response from the business agent)"
)

_SYSTEM_PROMPT = """You are role-playing a person calling a business on the phone. You are the CALLER, never the agent.

Your persona:
{persona}

Rules:
- Speak one short, natural spoken turn at a time. One or two sentences.
- You are on a phone call. No markdown, no lists, no stage directions.
- React only to what you actually heard. If what you heard was garbled or made no
  sense, respond the way a real person would -- ask them to repeat it, or say you
  didn't catch that.
- Do not be an assistant. Do not offer to help. You are the one who needs something.
- Stay in character even if the agent is unhelpful, wrong, or evasive.
- When your goal is met, or it is clear it will not be met, say a brief goodbye and
  then output {hangup} on its own line as the very last thing.
- Never output {hangup} on your first turn.

In the conversation below, the `user` messages are the business agent you called
and the `assistant` messages are you, the caller. Reply with your next spoken line
only, with no name or role prefix.

You called them. They are serving you. Never say things a receptionist would say --
never "how can I help you", "is there anything else", "thanks for calling", never
offer assistance or ask what they need. If your last turn reads like the business
talking, you have made a mistake.

Every reply you produce is the CALLER's next spoken line, and nothing else -- no
role prefix, no narration.
"""


@dataclass
class CallerPersona:
    """Persona context for one simulated caller, from the dataset's metadata."""

    goal: str = ""
    temperament: str = ""
    knowledge: str = ""
    voice: str | None = None
    max_turns: int = 6
    interrupt_after_ms: int | None = None
    # 0.0 = clean line. Degrades only what the agent hears, not the recording.
    line_noise: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> CallerPersona:
        metadata = metadata or {}
        persona = metadata.get("persona") or {}
        known = {"goal", "temperament", "knowledge"}
        return cls(
            goal=persona.get("goal", ""),
            temperament=persona.get("temperament", ""),
            knowledge=persona.get("knowledge", ""),
            voice=metadata.get("voice"),
            max_turns=int(metadata.get("max_turns", 6)),
            interrupt_after_ms=metadata.get("interrupt_after_ms"),
            line_noise=float(metadata.get("line_noise") or 0.0),
            extra={k: v for k, v in persona.items() if k not in known},
        )

    def describe(self) -> str:
        parts = []
        if self.goal:
            parts.append(f"- Why you are calling: {self.goal}")
        if self.temperament:
            parts.append(f"- How you come across: {self.temperament}")
        if self.knowledge:
            parts.append(f"- What you know going in: {self.knowledge}")
        parts += [f"- {k}: {v}" for k, v in self.extra.items()]
        return "\n".join(parts) or "- A member of the public calling the business."


@lru_cache(maxsize=1)
def _client() -> Any:
    from openai import AsyncOpenAI

    # Deliberately NOT wrap_openai. That records the request as a chat span whose
    # `assistant` output is the *caller's* next line -- so the trace's thread view
    # showed the agent saying what the caller was about to say. The call itself is
    # the conversation worth reading; this request is machinery behind it, so it is
    # traced as a plain span with the tokens logged by hand.
    return AsyncOpenAI(
        base_url=PROXY_BASE_URL,
        api_key=os.environ.get("BRAINTRUST_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
        # The proxy occasionally stalls under concurrency; a timeout that
        # retries beats failing the whole case on one slow request.
        timeout=45.0,
        max_retries=3,
    )


def _log_tokens(span: Any, response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    span.log(
        metrics={
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "tokens": usage.total_tokens,
        }
    )


# Phrases only the business side of a call would say. Used to catch the caller
# model slipping into receptionist voice, which the alternating-role format invites.
_BUSINESS_VOICE = (
    "how can i help",
    "how may i help",
    "is there anything else",
    "anything else i can",
    "thanks for calling",
    "thank you for calling",
    "how can i assist",
    "may i help you",
    "can i help you",
    "what can i do for you",
    "happy to help",
)


def _sounds_like_the_business(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _BUSINESS_VOICE)


class SimulatedCaller:
    """Holds the caller's side of the conversation and generates its next line."""

    def __init__(self, persona: CallerPersona) -> None:
        self._persona = persona
        self._system = _SYSTEM_PROMPT.format(
            persona=persona.describe(), hangup=HANGUP_TOKEN
        )
        # Roles as this *request* needs them, not as the call reads: the agent is
        # `user` and the caller is `assistant`, because the model is generating the
        # caller's line and a completion is always `assistant`. That also makes the
        # history end on a `user` message, which is the only portable shape --
        # ending on `assistant` is prefill, which Anthropic models reject outright.
        #
        # The readable view is not lost: `CallResult.as_output()["messages"]` maps
        # the call to caller=`user` / agent=`assistant`, and this request is traced
        # as a plain span with string input/output rather than a chat thread, so the
        # inversion is never rendered as a conversation.
        self._messages: list[dict[str, str]] = []
        self.done = False

    def heard(self, text: str) -> None:
        """Record what the caller heard the agent say (as transcribed from audio)."""
        said = text or "(you could not make out what they said)"
        self._messages.append({"role": "user", "content": said})

    def said(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})

    def _ensure_agent_context(self) -> None:
        """Keep caller-generation requests ending with what the caller heard."""
        if not self._messages or self._messages[-1]["role"] == "user":
            return
        self.heard(NO_COMPLETE_AGENT_REPLY)

    async def next_utterance(self) -> str:
        """Generate the caller's next spoken turn. Sets `done` on hang-up."""
        self._ensure_agent_context()
        heard = next(
            (m["content"] for m in reversed(self._messages) if m["role"] == "user"),
            "",
        )
        # Typed LLM so it reads as a model call and its tokens roll up, but with
        # plain string input/output rather than a messages array. The messages array
        # is what made the thread view render the caller's next line as an
        # `assistant` message -- the span type was never the problem.
        span = start_span(name="caller_generation", type=SpanTypeAttribute.LLM)
        response = await _client().chat.completions.create(
            model=CALLER_MODEL,
            temperature=0.7,
            max_tokens=120,
            # Only a leading system message. A trailing one held the persona
            # better, but a non-leading `system` role is rejected outright by
            # Anthropic's Messages API, so it made the caller model-specific. The
            # instruction lives in the leading prompt instead, which every provider
            # accepts -- changing CALLER_MODEL is then just changing the name.
            messages=[
                {"role": "system", "content": self._system},
                *self._messages,
            ],  # type: ignore[arg-type]
        )
        text = (response.choices[0].message.content or "").strip()
        if _sounds_like_the_business(text):
            # One correction attempt. A caller who slips into receptionist voice
            # corrupts the trajectory the judge grades, so it is worth a retry
            # rather than letting it into the transcript.
            retry = await _client().chat.completions.create(
                model=CALLER_MODEL,
                temperature=0.7,
                max_tokens=120,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{self._system}\n\nYour previous attempt was "
                            f"{text!r}, which is the business talking, not you. "
                            "You are the CALLER. Say what you want or need next, "
                            "in your own words."
                        ),
                    },
                    *self._messages,
                ],  # type: ignore[arg-type]
            )
            text = (retry.choices[0].message.content or "").strip() or text

        if HANGUP_TOKEN in text:
            self.done = True
            text = text.replace(HANGUP_TOKEN, "").strip()

        self.said(text or HANGUP_TOKEN)
        # input/output read the way the call reads: what the caller heard, and what
        # it decided to say back.
        span.log(
            input=heard,
            output=text,
            metadata={
                "model": CALLER_MODEL,
                "persona_goal": self._persona.goal,
                "hung_up": self.done,
            },
        )
        _log_tokens(span, response)
        span.end()
        return text
