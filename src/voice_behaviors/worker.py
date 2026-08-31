"""Worker lifecycle: prewarm, session build, and the `invoca_call` OTel root."""

import logging
from typing import Any

from livekit.agents import (
    AgentServer,
    AgentSession,
    APIConnectOptions,
    JobContext,
    JobProcess,
    RecordingOptions,
    TurnHandlingOptions,
    inference,
    room_io,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from .agent import Assistant
from .config import (
    AGENT_NAME,
    DEMO_GREETING,
    DEMO_SYSTEM_PROMPT,
    DEMO_TENANT_ID,
    DEMO_WORKFLOW_ID,
    LLM_MODEL,
    STT_MODEL,
    TTS_MODEL,
    TTS_SESSION_TIMEOUT,
    TTS_VOICE,
)
from .telemetry import get_tracer, setup_braintrust_telemetry

logger = logging.getLogger(__name__)


def prewarm(proc: JobProcess) -> None:
    """Pre-load heavy models and initialise Braintrust once per subprocess."""
    proc.userdata["vad"] = silero.VAD.load()
    # Init telemetry here (not per-session) so the tracer provider is registered
    # before the first AgentSession is built.
    setup_braintrust_telemetry()


def build_session(*, vad: Any, turn_detection: Any = None) -> AgentSession:
    """Build the AgentSession. Single source of truth for the agent's model config.

    Takes `vad` (and optionally `turn_detection`) rather than a JobContext so the
    eval harness can build the exact same session outside a worker -- see
    `voice_behaviors.simulation.runner`. Anything that changes here changes what
    the eval measures, which is the point.

    `turn_detection=None` leaves the key out of TurnHandlingOptions entirely, which
    is what makes the session auto-select (VAD endpointing) -- an explicit
    `turn_detection=None` would instead mean "no turn detection". The worker passes
    MultilingualModel(); it cannot be constructed off-job, since it resolves its
    inference executor from the job context.
    """
    turn_handling = TurnHandlingOptions(preemptive_generation={"enabled": True})
    if turn_detection is not None:
        turn_handling["turn_detection"] = turn_detection

    return AgentSession(
        stt=inference.STT(model=STT_MODEL, language="en"),
        llm=inference.LLM(model=LLM_MODEL),
        tts=inference.TTS(model=TTS_MODEL, voice=TTS_VOICE),
        vad=vad,
        turn_handling=turn_handling,
        conn_options=SessionConnectOptions(
            tts_conn_options=APIConnectOptions(
                timeout=TTS_SESSION_TIMEOUT, max_retry=3, retry_interval=1.0
            ),
        ),
    )


def _build_session(ctx: JobContext) -> AgentSession:
    return build_session(
        vad=ctx.proc.userdata["vad"],
        # MultilingualModel() needs a JobContext, so build it here (not in
        # prewarm). The ONNX model is already loaded in the inference subprocess
        # at worker startup.
        turn_detection=MultilingualModel(),
    )


async def _start_session(ctx: JobContext, session: AgentSession) -> None:
    await session.start(
        agent=Assistant(instructions=DEMO_SYSTEM_PROMPT, greeting=DEMO_GREETING),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
        # All off: the LiveKit Cloud observability path is not enabled on this
        # project, so any recording (transcript included) 401s on the session
        # report upload at teardown. Braintrust ingestion is via the span
        # processor in telemetry.py, not this -- so this only removes noise.
        record=RecordingOptions(
            audio=False, transcript=False, traces=False, logs=False
        ),
    )
    logger.info("session connected (room=%s)", ctx.room.name)


async def handle_session(ctx: JobContext) -> None:
    """Open the `invoca_call` OTel root, then build + start the session under it."""
    await ctx.connect()

    tracer = get_tracer()
    if tracer is None:
        # Tracing disabled -- run without the root wrapper.
        await _start_session(ctx, _build_session(ctx))
        return

    # (1) invoca_call as an OTel span. While it is the current span, LiveKit's
    #     `agent_session` span (and its agent_turn/user_turn descendants) nests
    #     under it -- the OTel-context equivalent of the native-SDK contract.
    with tracer.start_as_current_span("invoca_call") as call_span:
        call_span.set_attribute("lk.room_name", ctx.room.name)

        # (2) setup phases as child OTel spans. These close before session.start
        #     so invoca_call -- not setup -- is current when the session opens.
        with tracer.start_as_current_span("setup"):
            with tracer.start_as_current_span("setup.call_context_resolve"):
                call_span.set_attribute("workflow_id", DEMO_WORKFLOW_ID)
                call_span.set_attribute("tenant_id", DEMO_TENANT_ID)
                call_span.set_attribute("is_preview", False)
            with tracer.start_as_current_span("setup.session_build"):
                session = _build_session(ctx)

        # (3) Start the agent INSIDE invoca_call (setup spans already closed).
        await _start_session(ctx, session)


def create_server() -> AgentServer:
    """Build and return the configured AgentServer."""
    server = AgentServer()
    server.setup_fnc = prewarm
    # agent_name pins explicit dispatch routing. Drop it to auto-handle all rooms.
    server.rtc_session(agent_name=AGENT_NAME)(handle_session)
    return server
