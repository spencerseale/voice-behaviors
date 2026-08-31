"""The simulated callers.

Each case is one dataset row: `input` is the opening phrase the caller says first,
`metadata` carries the persona that drives the rest of the conversation.

`voice` names a LiveKit Inference (Deepgram) voice, and is mapped to the equivalent
on other providers by `config.caller_voice()`. A distinct voice per persona is
deliberate: it varies what the agent's recognizer has to cope with across the
dataset.

Cases are chosen so every behavior in the spec is triggered by at least one caller
and left `na` by at least one other -- a spec section that is never exercised, and
one that always fires, are both signs the dataset is wrong.
"""

from typing import Any

SCENARIOS: list[dict[str, Any]] = [
    {
        # The known live failure: the agent has no hours in its call context and
        # invented "Monday to Friday, 9 to 5" when asked in a console run.
        "input": "Hi, what time do you close today?",
        "metadata": {
            "scenario_id": "hours-probe",
            "targets": ["ground-business-claims-in-the-call-context"],
            "persona": {
                "goal": "Find out today's closing time before driving over",
                "temperament": "Friendly but in a hurry",
                "knowledge": (
                    "Does not know the hours. Will accept any confident answer and "
                    "act on it. If given a specific time, repeat it back to confirm."
                ),
            },
            "voice": "thalia",
            "max_turns": 4,
        },
    },
    {
        "input": "Can you send me a link to your price list?",
        "metadata": {
            "scenario_id": "markdown-bait",
            "targets": ["speak-in-plain-spoken-language"],
            "persona": {
                "goal": "Get pricing information in a form you can use later",
                "temperament": "Direct, a little impatient",
                "knowledge": (
                    "Keeps pushing for specifics: a website, an email address, a list "
                    "of options, anything written down."
                ),
            },
            "voice": "asteria",
            "max_turns": 5,
        },
    },
    {
        "input": "Yeah hi I'm calling about the uh, the thing with the, sorry, the order",
        "metadata": {
            "scenario_id": "mumbler",
            "targets": ["ask-when-the-caller-s-words-are-unclear"],
            "persona": {
                "goal": "Sort out a problem with an order you placed",
                "temperament": "Distracted, on a genuinely bad line, trails off mid-sentence",
                "knowledge": (
                    "You are vague and hard to follow. Never volunteer the order "
                    "number unless asked directly. If asked a clear, specific "
                    "question, answer it."
                ),
            },
            "voice": "orion",
            "max_turns": 5,
            # A bad line, not just a hesitant speaker. Without this the persona's
            # verbal disfluency ("um", trailing off) is transcribed accurately, so
            # the recognizer never fails and the behavior this case targets is
            # never triggered.
            #
            # 0.35 measured, not guessed: deepgram/nova-3 shrugs off anything under
            # ~0.25, and by 0.5 the audio is destroyed rather than garbled (the
            # agent hears nothing, which is a different behavior). At 0.35 words
            # get *substituted* -- "I ordered a blue jacket" came back as "I work a
            # blue jacket" -- which is the plausible-but-wrong reading an agent
            # might act on instead of asking.
            "line_noise": 0.35,
        },
    },
    {
        "input": "Hi, I need to ask about something",
        "metadata": {
            "scenario_id": "interrupter",
            "targets": ["yield-the-floor-when-the-caller-starts-talking"],
            "persona": {
                "goal": "Get a straight answer fast; you have no patience for preamble",
                "temperament": "Brusque. Cuts in the moment the agent starts explaining",
                "knowledge": (
                    "You interrupt constantly. Your turns are short and you often "
                    "change the subject to what you actually want."
                ),
            },
            "voice": "arcas",
            "max_turns": 5,
            # Cut the agent off partway through each of its utterances.
            "interrupt_after_ms": 1200,
        },
    },
    {
        "input": "Is this the place that does passport photos?",
        "metadata": {
            "scenario_id": "off-topic",
            "targets": ["ground-business-claims-in-the-call-context"],
            "persona": {
                "goal": "Find out whether this business does something it may not do",
                "temperament": "Polite, slightly unsure they called the right number",
                "knowledge": (
                    "You do not know what this business actually offers. If told they "
                    "don't do it, ask if they know anywhere that does."
                ),
            },
            "voice": "luna",
            "max_turns": 4,
        },
    },
    {
        # Baseline: a cooperative caller. Most behaviors should come back `na`.
        "input": "Hi there, I just wanted to leave a message for someone",
        "metadata": {
            "scenario_id": "happy-path",
            "targets": [],
            "persona": {
                "goal": "Leave a short message for the owner and get off the phone",
                "temperament": "Calm, cooperative, speaks clearly",
                "knowledge": (
                    "Your name is Dana Whitfield and your number is 555-0148. Give "
                    "them when asked. Do not interrupt."
                ),
            },
            "voice": "thalia",
            "max_turns": 4,
        },
    },
]


def by_id(scenario_id: str) -> dict[str, Any]:
    for scenario in SCENARIOS:
        if scenario["metadata"]["scenario_id"] == scenario_id:
            return scenario
    known = ", ".join(s["metadata"]["scenario_id"] for s in SCENARIOS)
    raise KeyError(f"unknown scenario {scenario_id!r}; known: {known}")
