"""Run one simulated voice call and return its trajectory.

Sequence per call:

    build_session(vad=...)            # the production session config
    session.input.audio  = caller line
    session.output.audio = caller ear
    session.start(agent=Assistant(...), record=False)   # no room

    agent greets  -> ear captures audio -> STT -> caller hears it
    caller speaks -> TTS -> line -> agent's own STT -> agent replies
    ... until hang-up, max turns, or the wall clock cap

The transcript is built only from audio: every AGENT line is a transcription of the
audio the agent actually produced, never its text output.
"""

from __future__ import annotations

import asyncio
import io as _io
import logging
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from braintrust import SpanTypeAttribute, start_span
from livekit import rtc
from livekit.agents import APIConnectOptions, inference
from livekit.agents import stt as stt_pkg
from livekit.agents.utils import http_context
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero

from ..agent import Assistant
from ..config import (
    CALLER_MODELS,
    CALLER_PROVIDER,
    DEMO_CALL_DATA,
    DEMO_GREETING,
    DEMO_SYSTEM_PROMPT,
    caller_voice,
)
from ..tracing import (
    CALL_CONTEXT_RESOLVE_SPAN,
    SESSION_BUILD_SPAN,
    SETUP_SPAN,
    CallTraceSpans,
)
from ..worker import build_session
from .audio import SAMPLE_RATE, CallerAudioInput, TranscribingAudioOutput
from .caller import CallerPersona, SimulatedCaller

logger = logging.getLogger(__name__)

# Hard caps so one wedged call can't stall the whole eval. The timeout is a
# backstop for a wedged call, not the normal way a call ends -- at ~20s per
# exchange a 5-turn persona needs well over two minutes, and cutting it off early
# truncates the trajectory the judge grades.
CALL_TIMEOUT = 210.0
AGENT_REPLY_TIMEOUT = 30.0
# How long the caller listens for its transcription of one agent utterance: longer
# for the first result, then a short tail for anything arriving just behind it.
STT_FIRST_TIMEOUT = 6.0
# The recognizer emits one final transcript per detected pause, so a two-sentence
# utterance arrives in pieces and we must wait past the first one. But this tail is
# paid on every single turn as pure dead air, so it is kept tight: the whole
# utterance is pushed at once, far faster than real time, so the remaining pieces
# land close behind the first.
STT_FOLLOWUP_TIMEOUT = 1.2
# How long a non-interrupting caller waits for the agent to actually stop talking,
# and how much continuous quiet counts as "stopped".
AGENT_QUIET_TIMEOUT = 15.0
AGENT_QUIET_GRACE = 0.6
# Silence after each caller utterance, so VAD endpointing sees speech stop.
TRAILING_SILENCE = 0.8

_VAD: Any = None


def _caller_models(persona: CallerPersona) -> tuple[Any, Any]:
    """Build the simulated caller's ear and voice for the configured provider.

    Both must be streaming-capable: the caller transcribes the agent *while it
    speaks* rather than after it finishes, which is what keeps the gap between
    turns short. OpenAI's STT needs `use_realtime=True` for that; without it
    `stream()` raises and the caller hears nothing.
    """
    models = CALLER_MODELS[CALLER_PROVIDER]
    voice = caller_voice(persona.voice)

    if CALLER_PROVIDER == "openai":
        return (
            lk_openai.STT(model=models["stt"], language="en", use_realtime=True),
            lk_openai.TTS(model=models["tts"], voice=voice),
        )

    return (
        inference.STT(model=models["stt"], language="en"),
        inference.TTS(model=models["tts"], voice=voice),
    )


def _vad() -> Any:
    """Load silero once per process; it's ~50MB and reused across cases."""
    global _VAD  # noqa: PLW0603
    if _VAD is None:
        _VAD = silero.VAD.load()
    return _VAD


@dataclass
class CallTurn:
    speaker: str  # "caller" | "agent" | "event"
    text: str
    interrupted: bool = False
    # For caller turns: what the agent's recognizer actually received, when that
    # differs from what was said. None when the line carried the words intact.
    misheard_as: str | None = None


@dataclass
class CallResult:
    transcript: str
    turns: list[CallTurn] = field(default_factory=list)
    audio: bytes | None = None
    ended_because: str = ""
    duration_s: float = 0.0
    # What this call consumed from LiveKit Inference, per model.
    usage: dict[str, Any] = field(default_factory=dict)
    # One entry per agent turn: what it said plus LiveKit's own latency metrics.
    agent_turns: list[dict[str, Any]] = field(default_factory=list)
    # Every spoken turn with absolute start/end epochs, used for output and for
    # fallback trace replay when a result was not logged while the call ran.
    spoken_turns: list[dict[str, Any]] = field(default_factory=list)
    # True when the simulator emitted native Braintrust turn spans as the call ran.
    trace_spans_logged: bool = False

    def as_output(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "turns": [
                {"speaker": t.speaker, "text": t.text, "interrupted": t.interrupted}
                for t in self.turns
            ],
            # The conversation in chat roles: the synthetic caller is the `user`,
            # the agent under test is the `assistant`. This is the view of the call
            # a reviewer reads, so it uses the roles the parties actually play --
            # not the roles of whichever model happened to generate each line.
            "messages": [
                {
                    "role": "user" if turn.speaker == "caller" else "assistant",
                    "content": turn.text,
                }
                for turn in self.turns
                if turn.speaker in ("caller", "agent")
            ],
            "ended_because": self.ended_because,
            "duration_s": round(self.duration_s, 1),
            "usage": self.usage,
        }


def call_context() -> str:
    """What the agent was told about this call -- the judge's grounding baseline."""
    facts = "\n".join(f"- {k}: {v}" for k, v in DEMO_CALL_DATA.items())
    return (
        "The agent was given only the following information about this call:\n"
        f"{facts}\n"
        "It was given no other business facts: no hours, prices, address, policies, "
        "inventory, or account details.\n\n"
        f"Its instructions were:\n{DEMO_SYSTEM_PROMPT}"
    )


@dataclass
class _Segment:
    """One side's audio, placed at the second it started on the call timeline."""

    at: float
    pcm: bytes
    rate: int
    speaker: str  # "caller" | "agent"


def _resample(pcm: bytes, rate: int) -> bytes:
    if rate == SAMPLE_RATE:
        return pcm
    resampler = rtc.AudioResampler(rate, SAMPLE_RATE, num_channels=1)
    frame = rtc.AudioFrame(
        data=pcm, sample_rate=rate, num_channels=1, samples_per_channel=len(pcm) // 2
    )
    return b"".join(bytes(f.data) for f in resampler.push(frame) + resampler.flush())


def _to_stereo_wav(segments: list[_Segment]) -> bytes:
    """Lay both sides onto one timeline: caller left, agent right.

    Stereo rather than a mono mix so the two voices stay separable, and placed by
    timestamp rather than concatenated so overlaps are preserved -- when the agent
    talks over the caller you can actually hear it, which is the thing the
    interruption behavior is about.
    """
    if not segments:
        return b""

    placed = [
        (seg, _resample(seg.pcm, seg.rate), max(0, int(seg.at * SAMPLE_RATE)))
        for seg in segments
    ]
    total = max(start + len(pcm) // 2 for _, pcm, start in placed)

    # int32 accumulator: two segments on the same side can touch, and summing in
    # int16 would wrap instead of clip.
    tracks = {
        "caller": np.zeros(total, dtype=np.int32),
        "agent": np.zeros(total, dtype=np.int32),
    }
    for seg, pcm, start in placed:
        samples = np.frombuffer(pcm, dtype=np.int16)
        tracks[seg.speaker][start : start + len(samples)] += samples

    stereo = np.stack(
        [
            np.clip(tracks["caller"], -32768, 32767).astype(np.int16),
            np.clip(tracks["agent"], -32768, 32767).astype(np.int16),
        ],
        axis=-1,
    )

    buf = _io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(stereo.tobytes())
    return buf.getvalue()


def measure_overlap(segments: list[_Segment]) -> tuple[float, int]:
    """Seconds of simultaneous speech, and how many episodes, from the timeline.

    Measured from the same segment timestamps that render the stereo audio, so the
    transcript footer and the recording can never disagree. This replaces counting
    hook events, which only saw the caller cutting in and was structurally blind to
    the reverse -- with preemptive generation the agent can start replying before
    the caller has stopped, and that overlap is the agent's fault, not the caller's.
    """
    spans = {"caller": [], "agent": []}
    for seg in segments:
        spans[seg.speaker].append((seg.at, seg.at + len(seg.pcm) / 2 / seg.rate))

    merged = {}
    for speaker, intervals in spans.items():
        out: list[list[float]] = []
        for start, end in sorted(intervals):
            if out and start <= out[-1][1]:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([start, end])
        merged[speaker] = out

    total, episodes = 0.0, 0
    for c_start, c_end in merged["caller"]:
        for a_start, a_end in merged["agent"]:
            span = min(c_end, a_end) - max(c_start, a_start)
            if span > 0.05:  # ignore sub-frame touching at boundaries
                total += span
                episodes += 1
    return round(total, 1), episodes


def render_transcript(
    turns: list[CallTurn],
    overlap_s: float = 0.0,
    episodes: int = 0,
    cutoffs: int = 0,
) -> str:
    """Render the trajectory, including whether the two sides ever overlapped.

    The footer is not decoration. Whether the caller began speaking while the agent
    was speaking is a fact about *timing*, and a plain alternating transcript cannot
    express it -- so a judge asked to grade interruption handling has no evidence and
    will guess. Stating the overlap count explicitly, including when it is zero, is
    what lets a behavior that never triggered come back `na` instead of `false`.
    """
    lines = []
    for turn in turns:
        if turn.speaker == "event":
            lines.append(f"[{turn.text}]")
        else:
            label = turn.speaker.upper()
            suffix = " [cut off mid-utterance]" if turn.interrupted else ""
            lines.append(f"{label}: {turn.text}{suffix}")
            if turn.misheard_as is not None:
                lines.append(
                    f"  [what the agent actually received: {turn.misheard_as!r} "
                    "-- the line garbled this turn]"
                )

    if episodes or cutoffs:
        lines.append(
            f"[call events: the caller and the agent were speaking at the same time "
            f"for {overlap_s}s across {episodes} episode(s); {cutoffs} agent "
            f"utterance(s) were cut off mid-delivery]"
        )
    else:
        lines.append(
            "[call events: no interruption occurred at any point in this call. The "
            "two never spoke at the same time, turns strictly alternated, and no "
            "agent utterance was cut off. Neither party ever had to yield the "
            "floor.]"
        )
    return "\n".join(lines)


class _CallRunner:
    def __init__(self, opening: str, persona: CallerPersona) -> None:
        self._opening = opening
        self._persona = persona
        self._caller = SimulatedCaller(persona)
        self._line = CallerAudioInput(noise=persona.line_noise)
        self._ear = TranscribingAudioOutput()
        self._ear.on_frame = self._overhear
        self._live_stt: Any = None
        self._turns: list[CallTurn] = []
        self._segments: list[_Segment] = []
        self._t0 = 0.0
        # Timing facts the transcript alone cannot carry; surfaced in its footer.
        self._overlaps = 0
        self._cutoffs = 0
        # Wall clock at call start; segment offsets are relative to it.
        self._epoch0 = 0.0
        self._session: Any = None
        self._trace = CallTraceSpans()
        # How many user-role messages the agent's recognizer had produced when the
        # last caller turn was spoken. New ones since then belong to that turn.
        self._heard_cursor = 0
        self._agent_history_cursor = 0
        self._caller_stt, self._caller_tts = _caller_models(persona)

    async def run(self) -> CallResult:
        # LiveKit plugins pull their aiohttp session from a job-scoped context.
        # Outside a worker there isn't one, so bind a process-local session for
        # the duration of the call -- this covers the agent's STT/LLM/TTS and the
        # caller's own TTS/STT alike.
        async with http_context.open():
            return await self._run()

    async def _run(self) -> CallResult:
        started = self._t0 = time.monotonic()
        self._epoch0 = time.time()
        with start_span(name=SETUP_SPAN, type=SpanTypeAttribute.TASK):
            with start_span(
                name=CALL_CONTEXT_RESOLVE_SPAN, type=SpanTypeAttribute.TASK
            ) as span:
                span.log(metadata={"call_context": call_context()})
            with start_span(name=SESSION_BUILD_SPAN, type=SpanTypeAttribute.TASK):
                session = self._session = build_session(vad=_vad())
        session.input.audio = self._line
        session.output.audio = self._ear
        self._line.start()
        self._trace.start_session(start_time=self._epoch0)

        ended = "completed"
        usage: dict[str, Any] = {}
        agent_turns: list[dict[str, Any]] = []
        try:
            await session.start(
                agent=Assistant(
                    instructions=DEMO_SYSTEM_PROMPT, greeting=DEMO_GREETING
                ),
                record=False,
            )
            # Hard outer cap: the per-wait deadlines below bound individual steps,
            # this bounds the call as a whole so one wedged case can't stall a run.
            ended = await asyncio.wait_for(self._converse(), timeout=CALL_TIMEOUT)
        except asyncio.TimeoutError:
            ended = "call timeout"
            self._turns.append(CallTurn("event", "call timed out"))
        finally:
            # In `finally`, not on one branch: a call that hit the timeout still
            # consumed inference, and those are exactly the calls whose cost you
            # want to see. Must run before aclose() tears the session down.
            usage = self._collect_usage(session)
            agent_turns = self._collect_agent_turns(session)
            self._trace.apply_agent_turn_metrics(agent_turns)

            with _suppress():
                await session.aclose()
            with _suppress():
                await self._line.aclose()
            with _suppress():
                await self._ear.aclose()
            # The recognizer is opened lazily while the agent speaks and normally
            # closed when its utterance is transcribed. A call that ends on a
            # timeout or a barge-in never reaches that close-out, so without this
            # every such call leaks an open socket -- which is what starts getting
            # the TTS handshake 429'd, even running one call at a time.
            self._ear.on_frame = None
            with _suppress():
                await self._close_live_stt()
            self._trace.end_session(end_time=time.time())

        return CallResult(
            transcript=render_transcript(
                self._turns, *measure_overlap(self._segments), self._cutoffs
            ),
            turns=self._turns,
            audio=_to_stereo_wav(self._segments) or None,
            ended_because=ended,
            duration_s=time.monotonic() - started,
            usage=usage,
            agent_turns=agent_turns,
            spoken_turns=self._spoken_turns(),
            trace_spans_logged=self._trace.spans_logged,
        )

    def _spoken_turns(self) -> list[dict[str, Any]]:
        """Each spoken turn with the absolute epoch it started and ended.

        Derived from the audio timeline, which is the only record of when each side
        actually spoke. Without real timestamps the fallback spans all land at
        the end of the case and read as though the conversation happened after
        every model call, rather than the model calls happening inside the
        conversation.
        """
        turns = []
        texts = {"caller": [], "agent": []}
        for turn in self._turns:
            if turn.speaker in texts:
                texts[turn.speaker].append(turn.text)

        counters = {"caller": 0, "agent": 0}
        for seg in sorted(self._segments, key=lambda s: s.at):
            said = texts[seg.speaker]
            index = counters[seg.speaker]
            counters[seg.speaker] += 1
            turns.append(
                {
                    "speaker": seg.speaker,
                    "text": said[index] if index < len(said) else "",
                    "start": self._epoch0 + seg.at,
                    "end": self._epoch0 + seg.at + len(seg.pcm) / 2 / seg.rate,
                }
            )
        return turns

    def _capture_heard(self) -> None:
        """Attribute the agent's new STT output to the caller turn just spoken.

        Read chronologically rather than reconstructed afterward: the agent's
        recognizer emits final transcripts on its own endpointing schedule, so one
        caller utterance can become two user messages or none. Pairing by index
        after the fact silently attributes one turn's audio to another, which is
        worse than no annotation -- it invents evidence.
        """
        if self._session is None:
            return
        try:
            heard = [
                (item.text_content or "").strip()
                for item in self._session.history.items
                if getattr(item, "type", None) == "message" and item.role == "user"
            ]
        except Exception as exc:
            logger.warning("could not read what the agent heard: %s", exc)
            return

        fresh = [h for h in heard[self._heard_cursor :] if h]
        self._heard_cursor = len(heard)
        if not fresh:
            return

        # The most recent caller turn is the one this audio came from.
        for turn in reversed(self._turns):
            if turn.speaker == "caller":
                combined = " ".join(fresh)
                if _materially_different(turn.text, combined):
                    turn.misheard_as = combined
                return

    def _collect_agent_turns(self, session: Any) -> list[dict[str, Any]]:
        """Per-turn text and latency, straight from the session's own history.

        LiveKit records these on each ChatMessage (`metrics_collected` is
        deprecated in favour of this). Replayed as spans in the eval so the
        agent's work is visible inside the experiment row rather than only as a
        separate trace in the project logs.
        """
        turns: list[dict[str, Any]] = []
        try:
            for item in session.history.items:
                if getattr(item, "type", None) != "message":
                    continue
                metrics = dict(getattr(item, "metrics", None) or {})
                turns.append(
                    {
                        "role": item.role,
                        "text": item.text_content or "",
                        "created_at": getattr(item, "created_at", None),
                        "metrics": {k: round(v, 3) for k, v in metrics.items()
                                    if isinstance(v, (int, float))},
                    }
                )
        except Exception as exc:  # reporting must never fail a call
            logger.warning("could not read session history: %s", exc)
        return turns

    def _next_agent_history_turn(self) -> dict[str, Any] | None:
        if self._session is None:
            return None
        assistant_turns = [
            turn
            for turn in self._collect_agent_turns(self._session)
            if turn["role"] == "assistant"
        ]
        if self._agent_history_cursor >= len(assistant_turns):
            return None
        turn = assistant_turns[self._agent_history_cursor]
        self._agent_history_cursor += 1
        return turn

    def _collect_usage(self, session: Any) -> dict[str, Any]:
        """Tally what this call actually consumed from LiveKit Inference.

        The agent's own STT/LLM/TTS are metered by the session. The simulated
        caller's TTS/STT are separate clients the session knows nothing about, so
        they are counted here from the audio we synthesized and sent for
        recognition -- they are billed the same way, and the simulation roughly
        doubles the speech spend of a real call.
        """
        usage: dict[str, Any] = {}
        try:
            for model_usage in session.usage.model_usage:
                entry = model_usage.model_dump()
                key = f"agent.{entry.get('type', 'usage')}.{entry.get('model', '?')}"
                usage[key] = {k: v for k, v in entry.items() if v not in (0, 0.0, "")}
        except Exception as exc:  # usage is reporting, never worth failing a call
            logger.warning("could not read session usage: %s", exc)

        caller_audio = sum(
            len(seg.pcm) / 2 / seg.rate for seg in self._segments if seg.speaker == "caller"
        )
        agent_audio = sum(
            len(seg.pcm) / 2 / seg.rate for seg in self._segments if seg.speaker == "agent"
        )
        usage["caller.tts_audio_s"] = round(caller_audio, 1)
        # The caller transcribes every second of agent speech it hears.
        usage["caller.stt_audio_s"] = round(agent_audio, 1)
        return usage

    async def _converse(self) -> str:
        deadline = time.monotonic() + CALL_TIMEOUT

        # The agent greets on_enter, before the caller says anything.
        await self._hear_agent(deadline)

        utterance = self._opening
        for turn in range(self._persona.max_turns):
            if not utterance:
                return "caller hung up"
            await self._speak(utterance)
            self._turns.append(CallTurn("caller", utterance))

            heard_ok = await self._hear_agent(deadline)
            # After the reply, not before: the agent's recognizer emits its final
            # transcript as part of producing that reply, so waiting until now is
            # what makes the attribution correct.
            self._capture_heard()
            if not heard_ok:
                return "agent went silent"
            if time.monotonic() > deadline:
                return "call timeout"
            if self._caller.done:
                return "caller hung up"

            utterance = await self._caller.next_utterance()
            if self._caller.done and not utterance:
                return "caller hung up"
            if turn == self._persona.max_turns - 1:
                return "max turns reached"

        return "max turns reached"

    def _overhear(self, frame: Any) -> None:  # noqa: D401
        """Feed the agent's audio to the caller's recognizer while it plays.

        The caller used to wait for the whole utterance and only then hand it over
        for transcription -- a serial round trip on every turn, and the single
        largest source of dead air between the agent finishing and the caller
        replying. Listening as it plays means the transcript is essentially ready
        the moment the agent stops, which is what a person on a phone does.
        """
        if self._live_stt is None:
            self._live_stt = self._caller_stt.stream(
                conn_options=APIConnectOptions(max_retry=3, timeout=20)
            )
        self._live_stt.push_frame(frame)

    async def _wait_until_agent_quiet(self) -> None:
        """Let the agent finish before the caller replies.

        The caller reacts to the first utterance it hears, but the agent may still
        be speaking -- a reply generated in two parts arrives as two utterances.
        Talking into that gap is the harness interrupting, not the caller, and it
        gets graded against the agent's yield-the-floor behavior. Interrupters
        skip this on purpose: talking over the agent is the point.
        """
        deadline = time.monotonic() + AGENT_QUIET_TIMEOUT
        quiet_since: float | None = None
        while time.monotonic() < deadline:
            if self._ear.speaking.is_set() or not self._ear.utterances.empty():
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= AGENT_QUIET_GRACE:
                return
            await asyncio.sleep(0.1)

    async def _speak(self, text: str) -> None:
        """Synthesize the caller's line and feed it onto the open line."""
        if not self._persona.interrupt_after_ms:
            await self._wait_until_agent_quiet()

        # Stream rather than collect(): pushing each chunk as it is synthesized
        # lets the caller start talking on the first chunk instead of waiting for
        # the whole utterance, which was several seconds of dead air per turn.
        stream = self._caller_tts.synthesize(
            text, conn_options=APIConnectOptions(max_retry=3, timeout=20)
        )
        chunks: list[bytes] = []
        rate = SAMPLE_RATE
        started_at: float | None = None
        try:
            async for ev in stream:
                if started_at is None:
                    started_at = time.monotonic() - self._t0
                    if self._ear.speaking.is_set():
                        self._turns.append(
                            CallTurn(
                                "event",
                                "the caller starts speaking while the agent is speaking",
                            )
                        )
                rate = ev.frame.sample_rate
                chunks.append(bytes(ev.frame.data))
                self._line.push_speech(ev.frame)
        finally:
            with _suppress():
                await stream.aclose()

        offset = started_at if started_at is not None else time.monotonic() - self._t0
        pcm = b"".join(chunks)
        self._segments.append(
            _Segment(at=offset, pcm=pcm, rate=rate, speaker="caller")
        )
        self._trace.log_user_turn(
            {
                "speaker": "caller",
                "text": text,
                "start": self._epoch0 + offset,
                "end": self._epoch0 + offset + len(pcm) / 2 / rate,
            }
        )
        await self._line.wait_until_spoken()
        # Trailing silence is what actually ends the caller's turn: VAD endpointing
        # fires on speech stopping, so without it the agent waits indefinitely.
        await asyncio.sleep(TRAILING_SILENCE)

    async def _hear_agent(self, deadline: float) -> bool:
        """Wait for the agent's next utterance and transcribe the audio it played."""
        remaining = min(AGENT_REPLY_TIMEOUT, max(0.0, deadline - time.monotonic()))
        if self._persona.interrupt_after_ms:
            # Barge-in: cut the agent off partway through instead of hearing it out.
            remaining = min(remaining, self._persona.interrupt_after_ms / 1000)

        try:
            pcm, interrupted = await asyncio.wait_for(
                self._ear.utterances.get(), timeout=remaining
            )
        except asyncio.TimeoutError:
            # Nothing was dequeued, so the close-out below never runs. Retire the
            # recognizer here: reusing it would splice this utterance onto the next
            # one and hand the caller a merged transcript.
            with _suppress():
                await self._close_live_stt()
            if self._persona.interrupt_after_ms:
                return True  # deliberately cutting in mid-utterance
            return False

        rate = self._ear.captured_rate
        offset = max(0.0, (time.monotonic() - self._t0) - (len(pcm) / 2 / rate))
        self._segments.append(_Segment(at=offset, pcm=pcm, rate=rate, speaker="agent"))
        if interrupted:
            self._cutoffs += 1
        text = await self._transcribe_live()
        self._turns.append(CallTurn("agent", text, interrupted=interrupted))
        self._trace.log_agent_turn(
            {
                "speaker": "agent",
                "text": text,
                "start": self._epoch0 + offset,
                "end": self._epoch0 + offset + len(pcm) / 2 / rate,
            },
            assistant_turn=self._next_agent_history_turn(),
        )
        self._caller.heard(text)
        return True

    async def _close_live_stt(self) -> None:
        """Drop the live recognizer without reading it (timeout / teardown paths)."""
        stream, self._live_stt = self._live_stt, None
        if stream is not None:
            with _suppress():
                stream.end_input()
            with _suppress():
                await stream.aclose()

    async def _transcribe_live(self) -> str:
        """Close out the recognizer that has been listening during playback."""
        stream, self._live_stt = self._live_stt, None
        if stream is None:
            return "(unintelligible)"

        parts: list[str] = []
        try:
            stream.end_input()
            while True:
                idle = STT_FOLLOWUP_TIMEOUT if parts else STT_FIRST_TIMEOUT
                try:
                    event = await asyncio.wait_for(stream.__anext__(), timeout=idle)
                except (asyncio.TimeoutError, StopAsyncIteration):
                    break
                if (
                    event.type == stt_pkg.SpeechEventType.FINAL_TRANSCRIPT
                    and event.alternatives
                ):
                    parts.append(event.alternatives[0].text.strip())
        except Exception as exc:  # a failed transcription is data, not a crash
            logger.warning("caller could not transcribe agent audio: %s", exc)
        finally:
            with _suppress():
                await stream.aclose()

        return " ".join(p for p in parts if p).strip() or "(unintelligible)"

class _suppress:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc is not None:
            logger.debug("ignored error during call teardown: %s", exc)
        return True


async def run_simulated_call(opening: str, persona: CallerPersona) -> CallResult:
    """Run one voice-to-voice call from an opening phrase and a persona."""
    return await _CallRunner(opening, persona).run()


def _materially_different(said: str, heard: str) -> bool:
    """True when a recognizer's output diverges enough to change the meaning.

    Word-level overlap rather than string equality: normal STT jitter (casing,
    filler words, punctuation) should not be reported as a mishearing, but a turn
    where a third of the content words went missing or changed should be.
    """
    import re

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2}

    said_words, heard_words = words(said), words(heard)
    if not said_words:
        return False
    overlap = len(said_words & heard_words) / len(said_words)
    return overlap < 0.7
