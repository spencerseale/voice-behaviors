"""Replay a finished call as Braintrust spans inside the current case.

The agent's own spans are OTel and only ever export to the project's logs, so
without this an experiment row shows the model calls made *around* the call but
nothing the agent under test actually said.

Two properties matter here and are easy to lose:

* **Span type.** The agent's turn is model output, so it is typed `llm` and rolls
  up with the other model work. The caller's turn is not the agent's model call --
  that is `caller_generation` -- so it stays a `task`.
* **Span time.** Each span carries the wall-clock moment its audio actually played.
  Replayed with default timestamps they all land at the end of the case and read as
  though the conversation happened after every model call rather than containing
  them.
"""

from __future__ import annotations

from typing import Any

from braintrust import SpanTypeAttribute, start_span


def log_call_spans(result: Any) -> int:
    """Emit one span per spoken turn. Returns how many were written."""
    per_turn_metrics = [
        turn["metrics"] for turn in result.agent_turns if turn["role"] == "assistant"
    ]
    agent_index = 0
    written = 0

    for index, turn in enumerate(result.spoken_turns, start=1):
        is_agent = turn["speaker"] == "agent"
        metrics = None
        if is_agent:
            if agent_index < len(per_turn_metrics):
                metrics = per_turn_metrics[agent_index] or None
            agent_index += 1

        span = start_span(
            name="agent_turn" if is_agent else "user_turn",
            # The agent's spoken turn is what its LLM produced, so it is an `llm`
            # span. Typing it `task` drops it out of the model-call view.
            type=SpanTypeAttribute.LLM if is_agent else SpanTypeAttribute.TASK,
            start_time=turn["start"],
        )
        span.log(
            input=None if is_agent else turn["text"],
            output=turn["text"] if is_agent else None,
            metadata={
                "turn_index": index,
                "speaker": turn["speaker"],
                "spoken_seconds": round(turn["end"] - turn["start"], 2),
            },
            metrics=metrics,
        )
        span.end(end_time=turn["end"])
        written += 1

    return written
