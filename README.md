# voice-behaviors

A LiveKit voice agent traced into Braintrust **over OpenTelemetry**, with a
masking hook on the export path.

## Why the OTel path

`braintrust==0.23.0`'s `auto_instrument()` produces flat, braintrust-native
spans: every per-turn operation (`eou_detection`, `stt_processing`,
`llm_request_run`, `tts_request`, `agent_speaking`) is parented directly onto one
long-lived `livekit_agent_session` span. LiveKit's own `agent_turn` / `user_turn`
spans exist, but they go to the OTel provider and never reach Braintrust — so
turn identity survives only as `speech_id` / `generation_id` metadata, not as
nesting.

Instead, this agent registers a `BraintrustSpanProcessor` on an OTel
`TracerProvider` and hands that provider to LiveKit via `set_tracer_provider`.
LiveKit's native OTel spans are exported to Braintrust as-is, so turns nest:

```
invoca_call                 ← OTel root, opened in handle_session
  setup
    setup.call_context_resolve
    setup.session_build
  agent_session
    user_turn / agent_turn  ← real turn boundaries
      stt / llm / tts
```

The root is an OTel span (not `braintrust.start_span`) because the two live in
different context systems — the OTel root is what LiveKit's `agent_session` span
parents under.

## Masking

The OTel path has no equivalent of `braintrust.set_masking_function`; that hook
only scrubs native-SDK spans, and `BraintrustSpanProcessor` exports spans
verbatim. `MaskingSpanProcessor` wraps the Braintrust processor and rewrites span
attributes and per-event attributes (where voice transcripts live) on end, before
the inner processor serializes them. `noop_masking_function` is a passthrough
stand-in for the production redactor — the registration point is what matters.

## Run it

```bash
cp .env.example .env    # fill in BRAINTRUST_API_KEY + LIVEKIT_* credentials
make agent              # interactive local session (mic/speaker)
```

`make agent` is `uv run voice-behaviors`, which defaults to LiveKit's `console`
mode. Other modes pass straight through:

```bash
uv run voice-behaviors console --text   # type instead of talk
uv run voice-behaviors --list-devices   # pick a mic/speaker
make dev                                # connect to LiveKit Cloud, await dispatch
```

Without `BRAINTRUST_API_KEY` the agent still runs; telemetry setup is skipped and
`handle_session` runs without the root span wrapper.

Traces land in the `voice-behaviors` Braintrust project. View them in
the trace **timeline**, not the summary metrics.

## Behavior eval

The agent is graded on its **conduct across a whole call**, following the
[Agent Behavior spec](https://agentbehavior.dev) and Braintrust's
[AgentBehavior cookbook](https://github.com/braintrustdata/braintrust-cookbook/tree/main/examples/AgentBehavior).

Behaviors live in `.agents/behaviors/voice-call-conduct/BEHAVIOR.md`. Each `##` section
is one behavior, graded independently by an LLM judge over the call trajectory:

| verdict | meaning | score |
| --- | --- | --- |
| `true` | the situation arose and the agent handled it correctly | 1 |
| `false` | the situation arose and the agent did not | 0 |
| `na` | the situation never arose in this call | `null` (excluded from the average) |

The spec is never shown to the agent — only to the judge — so the agent can't tailor
its conduct to a rubric it never sees.

### The trajectory is a real voice call

Each case runs a **voice-to-voice simulation**. A persona-driven caller speaks (TTS →
audio) into the agent's real STT, and hears the agent back by transcribing the audio the
agent's TTS actually produced. Both directions cross an audio boundary, so speech
failures — words the synthesizer mangles, a turn cut off by a barge-in — show up in the
transcript the judge grades.

```
caller LLM (persona) ─ TTS ─┐
      ▲                     ▼
      │           session.input.audio
   caller STT     AgentSession (real STT → LLM → TTS)
      │           session.output.audio
      └─────────────────────┘
```

`session.start()` only builds a RoomIO when the audio IO is unset, so assigning both
runs the whole session in-process — no room, no worker, no WebRTC. The session comes from
`build_session()`, the same function the production worker calls, so the eval measures
the deployed configuration.

### Run it

```bash
make dataset            # push the simulated callers to Braintrust
make sim CASE=hours-probe   # run one call locally, print the transcript
make eval               # grade every scenario against the spec
```

LiveKit Inference is the binding constraint. A full run is ~6 calls each holding four
or five Inference sockets, and a talkative case bills ~2 minutes of STT because the
recognizer streams the whole open line, silence included. Sustained iteration will hit a
project rate limit and surface as `429` on the TTS websocket handshake.

Runs are **serial** by default. Each call holds four LiveKit Inference sockets open (the
agent's STT/TTS plus the caller's) and the gateway 429s the TTS websocket handshake when
calls overlap — which fails the case outright, since a handshake rejection isn't covered
by request-level retry. Override with `EVAL_CONCURRENCY` if your quota allows.

Each case carries:

| | |
| --- | --- |
| `call_audio` | both sides, stereo — caller left, agent right, on a real timeline, so overlaps are audible |
| `messages` | the conversation in the roles the parties play: caller `user`, agent `assistant` |
| `livekit_usage` | what the call consumed per model — LLM tokens, TTS characters, STT audio seconds |
| `agent_turn` / `user_turn` spans | each spoken turn, stamped with the wall-clock time it actually happened so it interleaves with the model calls made inside it |
| `caller_generation` spans | the simulated caller's model call: typed `llm` with token metrics, but logged as plain heard/said strings rather than a messages array — a messages array makes the thread view render the caller's next line as an `assistant` message, i.e. the agent appearing to say what the caller is about to say |

### What does *not* happen

An eval run writes **nothing to the project's logs**. `setup_braintrust_telemetry()` is
deliberately not called here: it installs the `BraintrustSpanProcessor`, whose spans
export against its own `parent`, so every simulated call used to dump a LiveKit trace
into the logs and mix synthetic eval traffic with real agent traffic.

`BRAINTRUST_OTEL_COMPAT` is also deliberately unset. It swaps in an OTel-backed context
manager, and with it on the `wrap_openai` spans for the judge and caller never reach the
experiment — taking their token counts with them. It does not buy nesting either: the
agent's OTel spans export to project logs either way.

### Simulating a bad line

A persona can set `line_noise` (0.0-1.0) to degrade the audio the **agent** hears —
broadband hiss plus short dropouts. Only `mumbler` uses it, at `0.35`.

The recording keeps the clean copy on purpose: if the attachment were degraded too,
a reviewer could not tell a bad line from a bad persona. Caller turns the agent
received differently are annotated inline, captured after the agent replies (not
reconstructed afterward — the recognizer splits turns on its own endpointing
schedule, so pairing by index attributes one turn's audio to another):

```
CALLER: ...the thing with the, sorry, the order
  [what the agent actually received: "I'm calling about a a mister thirty four"]
AGENT: Are you looking for assistance with something specific for the mister thirty four?
```

Two things worth knowing before changing the value:

* **0.35 was measured, not chosen.** `deepgram/nova-3` shrugs off anything below
  ~0.25; by 0.5 the audio is destroyed rather than garbled, which tests a different
  behavior (hearing nothing vs. mishearing). 0.35 is where words get *substituted*,
  which is what makes an agent act on a wrong reading.
* **A degraded line makes VAD endpointing unreliable**, so `mumbler` also fails
  `yield-the-floor` — caused by the harness's noise, not by an agent defect in
  turn-taking. It is no longer a single-behavior probe.

### Known gaps

- **Transcription error lands on the agent's account.** Because the caller hears the
  agent by transcribing its audio, a word the recognizer mangles reads as the agent
  speaking unclearly. That is the honest cost of a voice-to-voice return path — it's how
  a real caller experiences a bad line — but a low `speak-in-plain-spoken-language` score
  should be checked against the attached audio before it's blamed on the agent.
- Turn-taking uses silero VAD endpointing, not the production `MultilingualModel`, which
  resolves its inference executor from a job context and can't be built off-job.
- Judged behaviors can only be as good as the evidence in the trajectory. Interruption
  grading depends on the `[call events: ...]` footer, which states measured overlap
  seconds — including when it is zero. Without it a judge has no timing information and
  invents overlap; with a vaguer wording it read clean alternation as "the agent yielded
  well" and scored 1 for a behavior that never fired.
- The worker path (dispatch, RoomIO, the `invoca_call` root span) isn't exercised; the
  eval case span takes that role.
- **Verdicts move between runs on identical code.** The caller runs at temperature 0.7
  and the agent and STT are non-deterministic, so every run is a different conversation.
  One run cannot tell you whether a change helped — use repeated trials before drawing a
  conclusion.
- **Never compare verdicts across behavior-spec versions.** A spec change invalidates
  prior verdicts the way a scorer change invalidates cross-experiment comparison; the
  version is recorded on each experiment as `behavior_spec_version`.
- Judge verdicts have not been validated against human labels — fine for iteration, not
  yet a release gate.

## Layout

```
.agents/behaviors/voice-call-conduct/BEHAVIOR.md   the graded behaviors
evals/voice_call_conduct.eval.py                   Eval: data, task, one scorer per behavior
scripts/seed_dataset.py                            push scenarios to Braintrust
src/voice_behaviors/
  cli.py         entrypoint; defaults the bare invocation to `console`
  worker.py      prewarm, build_session, invoca_call root, AgentServer
  agent.py       Assistant (greeting, seeded chat_ctx, custom llm_node)
  telemetry.py   BraintrustSpanProcessor registration + tracer accessor
  masking.py     MaskingSpanProcessor + no-op redactor stand-in
  config.py      hardcoded stand-ins for proprietary per-call config
  scenarios.py   the simulated callers (dataset rows)
  behaviors.py   BEHAVIOR.md parser -> judged sections
  judge.py       one-behavior LLM judge -> true/false/na
  simulation/    audio IO, persona caller, call runner
```
