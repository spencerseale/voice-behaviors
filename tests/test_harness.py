"""Tests for the eval harness's pure logic.

Everything here is deliberately offline: no LiveKit, no OpenAI, no Braintrust. The
harness's correctness bugs this session were all in pure functions -- a dropout
rate that truncated to zero, an overlap detector that only saw one direction, a
misheard-attribution that paired by index -- and every one of them survived
"it ran and produced a transcript". These pin the behavior those bugs violated.

    uv run pytest -q
"""

from __future__ import annotations

import asyncio
import inspect
import random
from types import SimpleNamespace

import numpy as np
import pytest

import voice_behaviors.tracing as call_tracing
from voice_behaviors import audio_empathy
from voice_behaviors.audio_empathy import (
    AudioEmpathyJudgment,
    _call_audio_bytes,
    _parse_audio_empathy_judgment,
)
from voice_behaviors.behaviors import load_behavior_spec
from voice_behaviors.judge import _parse_verdict, verdict_to_score
from voice_behaviors.simulation.audio import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    _degrade,
)
from voice_behaviors.simulation.caller import (
    CallerPersona,
    NO_COMPLETE_AGENT_REPLY,
    SimulatedCaller,
    _sounds_like_the_business,
)
from voice_behaviors.simulation.runner import (
    CallTurn,
    _materially_different,
    _Segment,
    measure_overlap,
    render_transcript,
)


def _tone(samples: int) -> bytes:
    return (np.sin(np.arange(samples) * 0.05) * 8000).astype(np.int16).tobytes()


# --- audio degradation ------------------------------------------------------ #


def test_degrade_is_noop_at_zero():
    pcm = _tone(SAMPLE_RATE)
    assert _degrade(pcm, 0.0, random.Random(1)) == pcm


def test_degrade_preserves_length():
    pcm = _tone(SAMPLE_RATE)
    assert len(_degrade(pcm, 0.35, random.Random(1))) == len(pcm)


def test_dropouts_survive_per_frame_application():
    """The bug this pins: `int(seconds * noise * 40)` is 0 for a 20ms frame.

    The real path degrades one 20ms frame at a time, so an integer dropout count
    silently never fired and only (easily denoised) hiss was applied. The rate has
    to survive being applied frame by frame.
    """
    rng = random.Random(1)
    frame = _tone(SAMPLES_PER_FRAME)
    zeroed = 0
    for _ in range(50):  # one second of real-path audio
        chunk = np.frombuffer(_degrade(frame, 0.35, rng), dtype=np.int16)
        zeroed += int((chunk == 0).sum())
    # Expect roughly 40*0.35 gaps/sec of 20-60ms; assert it is materially nonzero.
    assert zeroed > SAMPLE_RATE * 0.05, f"only {zeroed} samples dropped in 1s"


def test_per_frame_dropouts_are_capped_at_frame_width():
    """Per-frame degradation is weaker than whole-buffer, and that is inherent.

    A dropout width is clamped to the buffer it is applied to, so the real path
    (20ms frames) can only ever lose 20ms at a time while a whole-utterance pass
    picks 20-60ms gaps. Documented rather than "fixed": the frame path still
    degrades enough to break recognition at 0.35, which is what matters, and
    widening it would mean buffering the caller's speech instead of streaming it.
    """
    rng = random.Random(7)
    frame = _tone(SAMPLES_PER_FRAME)
    per_frame = sum(
        int((np.frombuffer(_degrade(frame, 0.35, rng), dtype=np.int16) == 0).sum())
        for _ in range(50)
    )
    whole = int(
        (
            np.frombuffer(
                _degrade(_tone(SAMPLE_RATE), 0.35, random.Random(7)), dtype=np.int16
            )
            == 0
        ).sum()
    )
    assert per_frame > 0 and whole > 0
    assert per_frame < whole, "per-frame is expected to be the weaker path"


# --- overlap measurement --------------------------------------------------- #


def _seg(speaker: str, at: float, seconds: float) -> _Segment:
    return _Segment(
        at=at, pcm=b"\x00\x00" * int(SAMPLE_RATE * seconds),
        rate=SAMPLE_RATE, speaker=speaker,
    )


def test_no_overlap_when_turns_alternate():
    segs = [_seg("agent", 0.0, 2.0), _seg("caller", 3.0, 2.0)]
    assert measure_overlap(segs) == (0.0, 0)


def test_overlap_measured_when_caller_cuts_in():
    segs = [_seg("agent", 0.0, 3.0), _seg("caller", 2.0, 3.0)]
    seconds, episodes = measure_overlap(segs)
    assert episodes == 1 and seconds == pytest.approx(1.0, abs=0.05)


def test_overlap_detected_in_both_directions():
    """The old detector only fired when the *caller* cut in.

    With preemptive generation the agent can start replying before the caller has
    stopped; that overlap is the agent's fault and must still be measured.
    """
    agent_first = measure_overlap([_seg("caller", 0.0, 3.0), _seg("agent", 2.0, 3.0)])
    caller_first = measure_overlap([_seg("agent", 0.0, 3.0), _seg("caller", 2.0, 3.0)])
    assert agent_first[1] == caller_first[1] == 1
    assert agent_first[0] == pytest.approx(caller_first[0], abs=0.05)


# --- transcript rendering -------------------------------------------------- #


def test_footer_states_zero_overlap_explicitly():
    """A judge with no timing information invents overlap, so silence is not enough."""
    text = render_transcript([CallTurn("caller", "hi")], 0.0, 0, 0)
    assert "no interruption occurred" in text


def test_footer_reports_measured_overlap():
    text = render_transcript([CallTurn("caller", "hi")], 4.2, 2, 1)
    assert "4.2s" in text and "2 episode" in text


def test_misheard_turn_is_annotated():
    turn = CallTurn("caller", "I ordered a blue jacket")
    turn.misheard_as = "I work a blue jacket"
    text = render_transcript([turn], 0.0, 0, 0)
    assert "what the agent actually received" in text
    assert "I work a blue jacket" in text


def test_roles_map_caller_to_user_and_agent_to_assistant():
    from voice_behaviors.simulation.runner import CallResult

    result = CallResult(
        transcript="t",
        turns=[CallTurn("agent", "hello"), CallTurn("caller", "hi"),
               CallTurn("event", "ignored")],
    )
    assert result.as_output()["messages"] == [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "hi"},
    ]


# --- mishearing detection -------------------------------------------------- #


@pytest.mark.parametrize(
    "said,heard,expected",
    [
        ("I ordered a blue jacket", "I ordered a blue jacket", False),
        ("I ordered a blue jacket", "i ordered a blue jacket.", False),
        ("what time do you close", "what time do you close today", False),
        ("I ordered a blue jacket", "I work a blue jacket", True),
        ("my order number is 4471", "my daughter never is far", True),
    ],
)
def test_materially_different(said, heard, expected):
    assert _materially_different(said, heard) is expected


# --- judge ----------------------------------------------------------------- #


def test_verdict_scores():
    assert verdict_to_score("true") == 1.0
    assert verdict_to_score("false") == 0.0
    assert verdict_to_score("na") is None  # excluded, not a failure


@pytest.mark.parametrize(
    "raw", ["not json at all", '{"verdict": "maybe"}', '{"verdict":', ""]
)
def test_unparseable_verdict_returns_none_not_na(raw):
    """`na` would be silently excluded from the average, hiding a broken judge."""
    assert _parse_verdict(raw) is None


def test_valid_verdict_parses():
    judgment = _parse_verdict('{"verdict": "false", "rationale": "because"}')
    assert judgment is not None
    assert judgment.verdict == "false" and judgment.rationale == "because"


# --- trace shape ----------------------------------------------------------- #


def test_replayed_call_spans_match_livekit_shape_and_timing(monkeypatch):
    events = []

    class _FakeSpan:
        def __init__(self, name: str | None) -> None:
            self.name = name

        def start_span(self, name=None, type=None, start_time=None, set_current=None):
            events.append(
                ("child_start", self.name, name, start_time, type, set_current)
            )
            return _FakeSpan(name)

        def set_current(self) -> None:
            events.append(("set_current", self.name))

        def unset_current(self) -> None:
            events.append(("unset_current", self.name))

        def log(self, **kwargs) -> None:
            events.append(("log", self.name, kwargs))

        def end(self, end_time=None) -> None:
            events.append(("end", self.name, end_time))

    def fake_start_span(name=None, type=None, start_time=None, set_current=None):
        events.append(("start", name, start_time, type, set_current))
        return _FakeSpan(name)

    monkeypatch.setattr(call_tracing, "start_span", fake_start_span)
    result = SimpleNamespace(
        spoken_turns=[
            {"speaker": "caller", "text": "hi", "start": 10.0, "end": 11.0},
            {"speaker": "agent", "text": "hello", "start": 12.0, "end": 13.5},
        ],
        agent_turns=[
            {"role": "assistant", "created_at": 11.8, "metrics": {"tokens": 3}},
        ],
    )

    assert call_tracing.log_call_spans(result) == 2

    starts = [event for event in events if event[0] in ("start", "child_start")]
    assert starts == [
        (
            "start",
            call_tracing.AGENT_SESSION_SPAN,
            10.0,
            call_tracing.SpanTypeAttribute.TASK,
            False,
        ),
        (
            "child_start",
            call_tracing.AGENT_SESSION_SPAN,
            call_tracing.USER_TURN_SPAN,
            10.0,
            call_tracing.SpanTypeAttribute.TASK,
            False,
        ),
        (
            "child_start",
            call_tracing.AGENT_SESSION_SPAN,
            call_tracing.AGENT_TURN_SPAN,
            11.8,
            call_tracing.SpanTypeAttribute.TASK,
            False,
        ),
        (
            "child_start",
            call_tracing.AGENT_TURN_SPAN,
            call_tracing.LLM_NODE_SPAN,
            11.8,
            call_tracing.SpanTypeAttribute.LLM,
            False,
        ),
    ]
    assert ("end", call_tracing.LLM_NODE_SPAN, 12.0) in events
    assert events[-1] == ("end", call_tracing.AGENT_SESSION_SPAN, 13.5)


# --- audio empathy scorer -------------------------------------------------- #


class _Attachment:
    def __init__(self, data: bytes) -> None:
        self.data = data


def test_audio_empathy_reads_wav_bytes_from_attachment():
    assert _call_audio_bytes({"call_audio": _Attachment(b"RIFF...WAVE")}) == b"RIFF...WAVE"


def test_audio_empathy_requires_call_audio():
    with pytest.raises(ValueError, match="call_audio"):
        _call_audio_bytes({})


def test_audio_empathy_parse_valid_json():
    judgment = _parse_audio_empathy_judgment(
        '{"label": "adequate", "evidence": "The agent stays calm."}'
    )
    assert judgment == AudioEmpathyJudgment(
        label="adequate", evidence="The agent stays calm."
    )


@pytest.mark.parametrize(
    "raw", ["not json", '{"label":"warm"}', '{"label":', ""]
)
def test_audio_empathy_rejects_unusable_judgments(raw):
    assert _parse_audio_empathy_judgment(raw) is None


def test_audio_empathy_scorer_does_not_accept_transcript_output():
    signature = inspect.signature(audio_empathy.score_audio_empathy)
    assert "output" not in signature.parameters


def test_audio_empathy_scorer_uses_audio_not_transcript(monkeypatch):
    calls: list[bytes] = []

    async def fake_judge(wav: bytes) -> AudioEmpathyJudgment:
        calls.append(wav)
        return AudioEmpathyJudgment(
            label="empathetic", evidence="The agent sounds patient."
        )

    monkeypatch.setattr(audio_empathy, "judge_audio_empathy", fake_judge)
    score = asyncio.run(
        audio_empathy.score_audio_empathy(
            input="opening",
            metadata={"call_audio": _Attachment(b"RIFF audio bytes")},
        )
    )

    assert calls == [b"RIFF audio bytes"]
    assert score.name == "audio_empathy"
    assert score.score == 1.0
    assert score.metadata["transcript_used"] is False


# --- persona / spec -------------------------------------------------------- #


def test_persona_reads_line_noise_and_interrupt():
    persona = CallerPersona.from_metadata(
        {"persona": {"goal": "g"}, "line_noise": 0.35, "interrupt_after_ms": 1200}
    )
    assert persona.line_noise == 0.35 and persona.interrupt_after_ms == 1200


def test_persona_defaults_to_clean_line():
    assert CallerPersona.from_metadata({}).line_noise == 0.0


def test_simulated_caller_adds_context_after_missed_agent_reply(monkeypatch):
    calls: list[list[dict[str, str]]] = []

    class _FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs["messages"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Sorry, could you repeat that?")
                    )
                ],
                usage=None,
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions())
    )
    monkeypatch.setattr("voice_behaviors.simulation.caller._client", lambda: fake_client)

    caller = SimulatedCaller(CallerPersona())
    caller.said("Hi, I need to ask about something.")

    utterance = asyncio.run(caller.next_utterance())

    assert utterance == "Sorry, could you repeat that?"
    assert calls[0][-1] == {
        "role": "user",
        "content": NO_COMPLETE_AGENT_REPLY,
    }


@pytest.mark.parametrize(
    "text,flagged",
    [
        ("Can I help you with something?", True),
        ("How can I help?", True),
        ("I need to know your hours", False),
        ("Um, I ordered a medium", False),
    ],
)
def test_receptionist_voice_detection(text, flagged):
    assert _sounds_like_the_business(text) is flagged


def test_spec_parses_into_judged_sections():
    spec = load_behavior_spec()
    assert len(spec.sections) >= 4
    # The H1 preamble is not a judged behavior.
    assert all(s.title != "Voice call conduct" for s in spec.sections)
    # Slugs are the score column names, so they must be unique and stable.
    assert len({s.slug for s in spec.sections}) == len(spec.sections)


def test_every_behavior_is_targeted_by_a_scenario():
    """A spec section no scenario triggers is a dataset gap, not a passing eval."""
    from voice_behaviors.scenarios import SCENARIOS

    slugs = {s.slug for s in load_behavior_spec().sections}
    targeted = {t for s in SCENARIOS for t in s["metadata"]["targets"]}
    assert not slugs - targeted, f"untargeted behaviors: {slugs - targeted}"
    assert not targeted - slugs, f"targets with no behavior: {targeted - slugs}"
