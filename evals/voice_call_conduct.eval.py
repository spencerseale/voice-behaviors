"""Grade the voice agent's call conduct against .agents/behaviors/voice-call-conduct.

    make eval          # or: bt eval --language python evals/

One scorer per `## ` section of the behavior spec. Each judges the whole call
trajectory and returns true/false/na, which maps to 1/0/None -- a behavior whose
situation never arose is excluded from the average rather than counted as a failure.

The spec is never shown to the agent. It reaches only the judge, so the agent cannot
tailor its conduct to a rubric it never sees.
"""

from __future__ import annotations

import os

from braintrust import (
    Attachment,
    Eval,
    Score,
    SpanTypeAttribute,
    init_dataset,
    start_span,
)
from dotenv import load_dotenv

load_dotenv()
# BRAINTRUST_OTEL_COMPAT is deliberately NOT set. It swaps in an OTel-backed
# context manager, and with it on the `wrap_openai` spans for the judge and the
# simulated caller never reach the experiment -- taking their token counts and
# costs with them. It also did not buy what it promised: the agent's LiveKit OTel
# spans still export to project logs either way, because BraintrustSpanProcessor
# resolves them against its own `parent`, not the current eval span.
#
# So the agent's per-call consumption is reported from `session.usage` instead
# (see simulation/runner.py:_collect_usage), which is metered by LiveKit itself.
os.environ.pop("BRAINTRUST_OTEL_COMPAT", None)

from voice_behaviors.behaviors import BehaviorSection, load_behavior_spec  # noqa: E402
from voice_behaviors.audio_empathy import (  # noqa: E402
    AUDIO_EMPATHY_MODEL,
    AUDIO_EMPATHY_SCORER_VERSION,
    score_audio_empathy,
)
from voice_behaviors.config import BRAINTRUST_PROJECT, LLM_MODEL  # noqa: E402
from voice_behaviors.judge import (  # noqa: E402
    JUDGE_MODEL,
    judge_behavior,
    verdict_to_score,
)
from voice_behaviors.simulation import CallerPersona, run_simulated_call  # noqa: E402
from voice_behaviors.simulation.runner import call_context  # noqa: E402
from voice_behaviors.tracing import JOB_ENTRYPOINT_SPAN, log_call_spans  # noqa: E402

SPEC = load_behavior_spec()
CONTEXT = call_context()

# Deliberately NOT calling setup_braintrust_telemetry() here. It installs the
# BraintrustSpanProcessor, whose spans export against its own `parent` -- so during
# an eval every simulated call dumped a LiveKit trace into the project's *logs*,
# mixing synthetic eval traffic in with real agent traffic. An experiment should
# leave the logs alone. What each call consumed is reported per case as
# `livekit_usage`, read from the session itself.


async def run_call(input: str, hooks) -> dict:  # noqa: A002 - Braintrust's arg name
    """Run one voice-to-voice call and hand back the trajectory to be judged."""
    persona = CallerPersona.from_metadata(hooks.metadata)

    with start_span(
        name=JOB_ENTRYPOINT_SPAN,
        type=SpanTypeAttribute.TASK,
    ) as root:
        result = await run_simulated_call(input, persona)

        call_metadata = {
            "entrypoint": "eval",
            "ended_because": result.ended_because,
            "duration_s": round(result.duration_s, 1),
            "livekit_usage": result.usage,
            "agent_model": LLM_MODEL,
            "call_context": CONTEXT,
        }
        root.log(input=input, output=result.as_output(), metadata=call_metadata)
        if not result.trace_spans_logged:
            log_call_spans(result)

    hooks.metadata["ended_because"] = result.ended_because
    hooks.metadata["duration_s"] = round(result.duration_s, 1)
    # LiveKit Inference consumption for this call, per model.
    hooks.metadata["livekit_usage"] = result.usage

    if result.audio:
        # Both sides of the call, stereo (caller left / agent right) on a real
        # timeline, so a reviewer can listen to what a verdict was based on rather
        # than trusting the transcription -- overlaps included.
        hooks.metadata["call_audio"] = Attachment(
            data=result.audio,
            filename=f"{hooks.metadata.get('scenario_id', 'call')}.wav",
            content_type="audio/wav",
        )
    return result.as_output()


def make_behavior_scorer(section: BehaviorSection):
    """One scorer per behavior. Judges only that behavior over the trajectory."""

    async def scorer(input, output, **_) -> Score:  # noqa: A002
        judgment = await judge_behavior(section, CONTEXT, output["transcript"])
        return Score(
            name=section.slug,
            score=verdict_to_score(judgment.verdict),
            metadata={
                "behavior": section.title,
                "verdict": judgment.verdict,
                "rationale": judgment.rationale,
            },
        )

    # Stable identity for the score column in the Braintrust UI.
    scorer.__name__ = section.slug
    return scorer


Eval(
    BRAINTRUST_PROJECT,
    data=init_dataset(project=BRAINTRUST_PROJECT, name="voice-call-scenarios"),
    task=run_call,
    scores=[make_behavior_scorer(section) for section in SPEC.sections]
    + [score_audio_empathy],
    metadata={
        "behavior_spec": SPEC.name,
        "behavior_spec_version": 2,
        "behavior_location": SPEC.location,
        "agent_model": LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "audio_empathy_model": AUDIO_EMPATHY_MODEL,
        "audio_empathy_scorer_version": AUDIO_EMPATHY_SCORER_VERSION,
        "simulation": "in-process voice-to-voice, VAD endpointing",
    },
    experiment_name=os.environ.get("EXPERIMENT_NAME"),
    # Each call holds ~4 LiveKit Inference sockets (the agent's STT/LLM/TTS plus
    # the caller's, when the caller is on LiveKit too). 4 concurrent calls is well
    # inside a 20-connection allowance while leaving room for the retry storm a
    # slow socket can cause -- and the whole 6-case dataset finishes in two waves.
    # The limit that bites first is the TTS websocket handshake, not the LLM: a
    # handshake 429 is not covered by request-level retry and fails the case.
    max_concurrency=int(os.environ.get("EVAL_CONCURRENCY", "4")),
)
