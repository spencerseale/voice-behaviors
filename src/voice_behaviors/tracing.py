"""Shared Braintrust span names and helpers for voice-call traces.

LiveKit worker runs emit real OpenTelemetry spans under ``job_entrypoint``. Eval
and sample runs deliberately do not install the OTel exporter, so the simulator
emits native Braintrust spans with the same public span names.

Two properties matter here and are easy to lose:

* **Span shape.** LiveKit's visible model work is nested as
  `agent_session -> agent_turn -> llm_node`; eval traces mirror that rather than
  collapsing the spoken reply into a leaf LLM span.
* **Span time.** Each span carries the wall-clock moment its audio actually played.
  Logged with default timestamps after teardown they land at the end of the case
  and read as though the conversation happened after every model call rather than
  containing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from braintrust import SpanTypeAttribute, start_span

JOB_ENTRYPOINT_SPAN = "job_entrypoint"
AGENT_SESSION_SPAN = "agent_session"
SETUP_SPAN = "setup"
CALL_CONTEXT_RESOLVE_SPAN = "setup.call_context_resolve"
SESSION_BUILD_SPAN = "setup.session_build"
AGENT_TURN_SPAN = "agent_turn"
USER_TURN_SPAN = "user_turn"
LLM_NODE_SPAN = "llm_node"


@dataclass
class _LoggedAgentTurn:
    turn_span: Any
    llm_span: Any


def _call_time_bounds(result: Any) -> tuple[float | None, float | None]:
    spoken_turns = list(getattr(result, "spoken_turns", []) or [])
    if not spoken_turns:
        return None, None
    return (
        min(turn["start"] for turn in spoken_turns),
        max(turn["end"] for turn in spoken_turns),
    )


def _turn_metrics(turn: dict[str, Any] | None) -> dict[str, Any]:
    if not turn:
        return {}
    metrics = turn.get("metrics") or {}
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


def _turn_created_at(turn: dict[str, Any] | None) -> float | None:
    if not turn:
        return None
    created_at = turn.get("created_at")
    return created_at if isinstance(created_at, (int, float)) else None


def _agent_lifecycle_start(
    turn_start: float,
    assistant_turn: dict[str, Any] | None,
) -> float:
    """Best observed start for the agent response lifecycle.

    LiveKit's `agent_turn` starts before audio playout and contains `llm_node`,
    TTS, and forwarding. Native eval/sample spans do not get the provider OTel
    subspans, so use the chat item's `created_at` when it is available and falls
    before speech.
    """
    created_at = _turn_created_at(assistant_turn)
    if created_at is not None:
        return min(created_at, turn_start)
    return turn_start


class CallTraceSpans:
    """Emit the eval/sample call tree as the simulator runs.

    The helper keeps `agent_session` open without making it the global current
    span. That lets simulator machinery such as `caller_generation` remain a
    direct child of the surrounding `job_entrypoint`, while spoken turns are
    explicitly parented under `agent_session` like the LiveKit OTel trace.
    """

    def __init__(self) -> None:
        self._session_span: Any | None = None
        self._agent_turns: list[_LoggedAgentTurn] = []
        self._turn_index = 0

    @property
    def spans_logged(self) -> bool:
        return bool(
            self._session_span is not None and getattr(self._session_span, "id", "")
        )

    def start_session(self, *, start_time: float | None = None) -> None:
        if self._session_span is not None:
            return
        self._session_span = start_span(
            name=AGENT_SESSION_SPAN,
            type=SpanTypeAttribute.TASK,
            start_time=start_time,
            set_current=False,
        )
        self._session_span.log(
            metadata={
                "span_source": "simulation_harness",
            }
        )

    def end_session(self, *, end_time: float | None = None) -> None:
        if self._session_span is None:
            return
        self._session_span.end(end_time=end_time)

    def log_user_turn(self, turn: dict[str, Any]) -> None:
        self._turn_index += 1
        span = self._start_turn_span(
            name=USER_TURN_SPAN,
            type=SpanTypeAttribute.TASK,
            start_time=turn["start"],
        )
        span.log(
            input=turn["text"],
            metadata=self._turn_metadata(turn, turn_index=self._turn_index),
        )
        span.end(end_time=turn["end"])

    def log_agent_turn(
        self,
        turn: dict[str, Any],
        *,
        assistant_turn: dict[str, Any] | None = None,
    ) -> None:
        self._turn_index += 1
        turn_start = _agent_lifecycle_start(turn["start"], assistant_turn)
        turn_span = self._start_turn_span(
            name=AGENT_TURN_SPAN,
            type=SpanTypeAttribute.TASK,
            start_time=turn_start,
        )
        turn_span.log(
            output=turn["text"],
            metadata=self._turn_metadata(
                turn,
                turn_index=self._turn_index,
                assistant_turn=assistant_turn,
            ),
            metrics=_turn_metrics(assistant_turn) or None,
        )

        llm_span = turn_span.start_span(
            name=LLM_NODE_SPAN,
            type=SpanTypeAttribute.LLM,
            start_time=turn_start,
            set_current=False,
        )
        llm_span.log(
            input=None,
            output=turn["text"],
            metadata={
                "span_source": "simulation_harness",
                "speaker": turn["speaker"],
                "model": _agent_model(assistant_turn),
            },
            metrics=_turn_metrics(assistant_turn) or None,
        )
        llm_span.end(
            end_time=turn["start"] if turn_start < turn["start"] else turn["end"]
        )
        turn_span.end(end_time=turn["end"])
        self._agent_turns.append(
            _LoggedAgentTurn(turn_span=turn_span, llm_span=llm_span)
        )

    def apply_agent_turn_metrics(self, agent_turns: list[dict[str, Any]]) -> None:
        """Merge late LiveKit message metrics into already-created turn spans."""
        assistant_turns = [
            turn for turn in agent_turns if turn.get("role") == "assistant"
        ]
        for logged, assistant_turn in zip(self._agent_turns, assistant_turns):
            metrics = _turn_metrics(assistant_turn)
            metadata = {"model": _agent_model(assistant_turn)}
            if not metrics and metadata["model"] is None:
                continue
            logged.turn_span.log(metrics=metrics or None, metadata=metadata)
            logged.llm_span.log(metrics=metrics or None, metadata=metadata)

    def _start_turn_span(
        self,
        *,
        name: str,
        type: SpanTypeAttribute,
        start_time: float,
    ) -> Any:
        if self._session_span is None:
            self.start_session(start_time=start_time)
        assert self._session_span is not None
        return self._session_span.start_span(
            name=name,
            type=type,
            start_time=start_time,
            set_current=False,
        )

    @staticmethod
    def _turn_metadata(
        turn: dict[str, Any],
        *,
        turn_index: int,
        assistant_turn: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "turn_index": turn_index,
            "speaker": turn["speaker"],
            "spoken_seconds": round(turn["end"] - turn["start"], 2),
            "span_source": "simulation_harness",
            "model": _agent_model(assistant_turn),
        }


def _agent_model(assistant_turn: dict[str, Any] | None) -> str | None:
    metrics = (assistant_turn or {}).get("metrics") or {}
    metadata = metrics.get("llm_metadata") if isinstance(metrics, dict) else None
    if isinstance(metadata, dict):
        return metadata.get("model_name")
    return None


def log_call_spans(result: Any) -> int:
    """Emit an ``agent_session`` subtree for an already-finished call.

    This is the fallback for static samples and externally supplied results. The
    simulator path emits spans while the call runs, so those rows are created in
    conversational order instead of all appearing at teardown.
    """
    start_time, end_time = _call_time_bounds(result)
    trace = CallTraceSpans()
    trace.start_session(start_time=start_time)
    written = _log_spoken_turn_spans(trace, result)
    trace.apply_agent_turn_metrics(list(getattr(result, "agent_turns", []) or []))
    trace.end_session(end_time=end_time)
    return written


def _log_spoken_turn_spans(trace: CallTraceSpans, result: Any) -> int:
    """Emit one span per spoken turn. Returns how many were written."""
    assistant_turns = [
        turn for turn in result.agent_turns if turn.get("role") == "assistant"
    ]
    agent_index = 0
    written = 0

    for turn in result.spoken_turns:
        is_agent = turn["speaker"] == "agent"
        assistant_turn = None
        if is_agent:
            if agent_index < len(assistant_turns):
                assistant_turn = assistant_turns[agent_index]
            agent_index += 1

        if is_agent:
            trace.log_agent_turn(turn, assistant_turn=assistant_turn)
        else:
            trace.log_user_turn(turn)
        written += 1

    return written
