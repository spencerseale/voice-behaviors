"""Voice-to-voice call simulation used to produce eval trajectories.

A persona-driven caller speaks (TTS -> audio frames) into the agent's real STT, and
hears the agent back by transcribing the audio the agent's TTS actually produced.
Both directions cross an audio boundary, so speech-specific failures are observable.

The session is built with `voice_behaviors.worker.build_session`, the same function
the production worker uses, so the eval measures the deployed configuration.
"""

from .audio import CallerAudioInput, TranscribingAudioOutput
from .caller import CallerPersona, SimulatedCaller
from .runner import CallResult, CallTurn, run_simulated_call

__all__ = [
    "CallResult",
    "CallTurn",
    "CallerAudioInput",
    "CallerPersona",
    "SimulatedCaller",
    "TranscribingAudioOutput",
    "run_simulated_call",
]
