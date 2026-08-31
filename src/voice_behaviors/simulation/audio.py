"""Audio IO that stands in for a phone line.

`session.start()` only creates a default RoomIO when `input.audio`/`output.audio`
are unset, so assigning these two makes the AgentSession run entirely in-process --
the same seam console mode uses for a local mic.

Both classes deliberately run in **real time**: the caller's speech is fed at wall
clock pace and the agent's speech is "played" at wall clock pace. Barge-in only
means something if the agent is genuinely mid-utterance when the caller starts
talking, so faking instantaneous playback would make one of the behaviors
unmeasurable.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import numpy as np

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice import io

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000


def silence_frame(samples: int = SAMPLES_PER_FRAME) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=SAMPLE_RATE,
        num_channels=NUM_CHANNELS,
        samples_per_channel=samples,
    )


def _degrade(pcm: bytes, noise: float, rng: random.Random) -> bytes:
    """Add hiss and occasional dropouts -- a bad phone line, not a quiet one.

    `noise` is roughly "fraction of full scale", so 0.0 is a clean line and ~0.05
    is audibly rough. Two effects, because hiss alone is easy for a modern
    recognizer to denoise around:

      * broadband noise, which lowers overall confidence
      * short dropouts, which delete whole phonemes and are what actually makes a
        recognizer emit the wrong word rather than no word

    Applied to the caller's audio only, and only on the way to the agent -- see
    CallerAudioInput.noise for why the recording keeps the clean copy.
    """
    if noise <= 0:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
    if samples.size == 0:
        return pcm

    hiss = np.array(
        [int(rng.gauss(0, noise * 6000)) for _ in range(samples.size)], dtype=np.int32
    )
    degraded = samples + hiss

    # Dropouts: ~40*noise gaps per second of audio. Expressed as an expected rate
    # and rounded stochastically, because this is applied per 20ms frame -- a plain
    # int() truncation gives int(0.02 * noise * 40) == 0 for every frame, so the
    # dropouts silently never happen and only the (easily denoised) hiss remains.
    seconds = samples.size / SAMPLE_RATE
    expected = seconds * noise * 40
    count = int(expected) + (1 if rng.random() < (expected - int(expected)) else 0)
    for _ in range(count):
        width = min(
            samples.size, rng.randint(SAMPLE_RATE // 50, SAMPLE_RATE // 16)
        )
        start = rng.randint(0, max(0, samples.size - width))
        degraded[start : start + width] = 0

    return np.clip(degraded, -32768, 32767).astype(np.int16).tobytes()


class CallerAudioInput(io.AudioInput):
    """An always-open line carrying the simulated caller's voice.

    Feeds silence continuously and splices in speech when the caller talks. The
    continuous stream matters: VAD endpointing decides the caller's turn ended by
    observing speech *stop*, so a stream that simply goes absent between utterances
    would never produce an end-of-turn.

    `noise` degrades what the agent hears without touching what the harness
    records. That asymmetry is deliberate: the recording should stay listenable so
    a reviewer can confirm the caller really did say the words the agent
    mis-transcribed, rather than being unable to tell a bad line from a bad
    persona.
    """

    def __init__(self, noise: float = 0.0) -> None:
        super().__init__(label="SimulatedCaller")
        self.noise = noise
        # Seeded so a case degrades the same way run to run; the conversation is
        # already non-deterministic without adding unseeded audio noise on top.
        self._rng = random.Random(1729)
        self._ch: utils.aio.Chan[rtc.AudioFrame] = utils.aio.Chan()
        self._speech: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue()
        self._attached = True
        self._pump_task: asyncio.Task[None] | None = None
        self._speaking = asyncio.Event()

    def start(self) -> None:
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())

    async def aclose(self) -> None:
        if self._pump_task is not None:
            await utils.aio.cancel_and_wait(self._pump_task)
            self._pump_task = None
        self._ch.close()

    async def _pump(self) -> None:
        """Emit one frame every FRAME_MS: queued speech if any, else silence."""
        period = FRAME_MS / 1000
        next_at = asyncio.get_running_loop().time()
        while True:
            next_at += period
            try:
                frame = self._speech.get_nowait()
                self._speaking.set()
            except asyncio.QueueEmpty:
                frame = silence_frame()
                self._speaking.clear()
            if self._attached:
                self._ch.send_nowait(frame)
            delay = next_at - asyncio.get_running_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # Fell behind (slow STT/CPU); resync rather than burst-sending
                # frames faster than real time, which would break VAD timing.
                next_at = asyncio.get_running_loop().time()

    def push_speech(self, frame: rtc.AudioFrame) -> None:
        """Queue one frame of caller speech, resampled to the line's format."""
        for chunk in _reframe(frame):
            if self.noise > 0:
                chunk = rtc.AudioFrame(
                    data=_degrade(bytes(chunk.data), self.noise, self._rng),
                    sample_rate=chunk.sample_rate,
                    num_channels=chunk.num_channels,
                    samples_per_channel=chunk.samples_per_channel,
                )
            self._speech.put_nowait(chunk)

    async def wait_until_spoken(self) -> None:
        """Return once every queued speech frame has been fed onto the line."""
        while not self._speech.empty():
            await asyncio.sleep(FRAME_MS / 1000)

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._ch.__anext__()

    def on_attached(self) -> None:
        self._attached = True

    def on_detached(self) -> None:
        self._attached = False


def _reframe(frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
    """Split an arbitrary-length frame into FRAME_MS frames at the line rate."""
    if frame.sample_rate != SAMPLE_RATE:
        resampler = rtc.AudioResampler(frame.sample_rate, SAMPLE_RATE, num_channels=1)
        frames = resampler.push(frame) + resampler.flush()
    else:
        frames = [frame]

    data = b"".join(bytes(f.data) for f in frames)
    stride = SAMPLES_PER_FRAME * 2
    out: list[rtc.AudioFrame] = []
    for offset in range(0, len(data), stride):
        chunk = data[offset : offset + stride]
        if len(chunk) < stride:
            chunk = chunk + b"\x00" * (stride - len(chunk))
        out.append(
            rtc.AudioFrame(
                data=chunk,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLES_PER_FRAME,
            )
        )
    return out


class TranscribingAudioOutput(io.AudioOutput):
    """Collects the agent's synthesized speech, one utterance at a time.

    Buffers frames as they arrive, plays them out at real time, and publishes a
    completed utterance on `flush()` (natural end) or `clear_buffer()`
    (interrupted). Callers pick utterances up from `utterances` and transcribe
    them -- the harness never reads the agent's text output, so what the caller
    reacts to is only ever what the agent actually said out loud.
    """

    def __init__(self) -> None:
        super().__init__(
            label="SimulatedCallerEar",
            next_in_chain=None,
            sample_rate=SAMPLE_RATE,
            # pause=True so the session's resume_false_interruption path stays
            # live: when it decides an interruption was spurious it pauses rather
            # than cancelling, and an output that can't pause silently disables
            # that -- which would change the very behavior we grade.
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self.utterances: asyncio.Queue[tuple[bytes, bool]] = asyncio.Queue()
        self._buf = bytearray()
        self._played = 0.0
        self._playing: asyncio.Task[None] | None = None
        self._pending = asyncio.Queue[rtc.AudioFrame]()
        self._flushed = False
        self.speaking = asyncio.Event()
        self._unpaused = asyncio.Event()
        self._unpaused.set()
        # A segment is "open" from the first captured frame until we report it
        # finished. on_playback_finished must fire exactly once per segment the
        # session captured, or the session logs a mismatch and its speech
        # bookkeeping drifts.
        self._segment_open = False
        self._rate = SAMPLE_RATE
        # Set by the runner to receive frames *as they play*, so the caller's
        # recognizer hears the agent in real time instead of the whole utterance
        # being handed over after it finishes.
        self.on_frame: Any = None

    @property
    def captured_rate(self) -> int:
        """Sample rate of the frames the session actually sent us."""
        return self._rate

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self._flushed = False
        self._segment_open = True
        self._rate = frame.sample_rate
        self.speaking.set()
        self._pending.put_nowait(frame)
        if self._playing is None:
            self._playing = asyncio.create_task(self._playout())

    def flush(self) -> None:
        super().flush()
        self._flushed = True

    def clear_buffer(self) -> None:
        """The session interrupted this utterance -- publish what was heard."""
        while not self._pending.empty():
            self._pending.get_nowait()
        self._finish(interrupted=True)

    def pause(self) -> None:
        super().pause()
        self._unpaused.clear()

    def resume(self) -> None:
        super().resume()
        self._unpaused.set()

    async def _playout(self) -> None:
        """Drain queued frames at wall clock pace, then close out the utterance."""
        try:
            while True:
                await self._unpaused.wait()
                try:
                    frame = await asyncio.wait_for(self._pending.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    # No more audio arriving. If the session flushed, the
                    # utterance is over; otherwise the TTS is still catching up.
                    if self._flushed:
                        self._finish(interrupted=False)
                        return
                    continue
                self._buf += bytes(frame.data)
                self._played += frame.duration
                if self.on_frame is not None:
                    try:
                        self.on_frame(frame)
                    except Exception:  # listening must never break playback
                        logger.debug("on_frame hook failed", exc_info=True)
                await asyncio.sleep(frame.duration)
        except asyncio.CancelledError:
            raise

    def _finish(self, *, interrupted: bool) -> None:
        if not self._segment_open:
            return  # nothing was captured; there is no segment to report
        self._segment_open = False

        if self._playing is not None:
            self._playing.cancel()
            self._playing = None
        audio, self._buf = bytes(self._buf), bytearray()
        played, self._played = self._played, 0.0
        self._flushed = False
        self.speaking.clear()
        if audio:
            self.utterances.put_nowait((audio, interrupted))
        # Tell the session the audio finished playing; without this it waits
        # forever on the speech handle and the call never advances.
        self.on_playback_finished(playback_position=played, interrupted=interrupted)

    async def aclose(self) -> None:
        if self._playing is not None:
            await utils.aio.cancel_and_wait(self._playing)
            self._playing = None
