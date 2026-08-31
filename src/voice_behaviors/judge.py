"""LLM judge: grade one call trajectory against one behavior.

Python port of the cookbook's `judge.ts`. Each behavior gets exactly one of three
grades over a trajectory:

    true  -> the behavior's trigger fired and the agent did the right thing
    false -> the trigger fired and the agent did not
    na    -> the trigger did not fire, or the trace cannot decide

`na` maps to a null Braintrust score so it is excluded from the average instead of
counting as a failure.

The judge is the ONLY place the behavior spec is read. It is never shown to the
agent, so the agent cannot tailor its conduct to a spec it never sees.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from braintrust import wrap_openai

from .behaviors import BehaviorSection

logger = logging.getLogger(__name__)

Verdict = Literal["true", "false", "na"]

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
# Braintrust gateway. Authentication is BRAINTRUST_API_KEY, but the gateway then
# calls the provider using the AI-provider credentials configured on the org (or
# project) -- so a healthy Braintrust key is necessary but not sufficient. A 401
# naming an `sk-proj-...` key is the provider credential, not yours; the
# `x-bt-error-origin` response header says which layer failed.
PROXY_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://gateway.braintrust.dev")


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a careful evaluator judging a voice agent's call trajectory against a "
    "single behavior. You ground every verdict in evidence from the transcript."
)


@dataclass(frozen=True)
class BehaviorJudgment:
    verdict: Verdict
    rationale: str


@lru_cache(maxsize=1)
def _client() -> Any:
    from openai import AsyncOpenAI

    return wrap_openai(
        AsyncOpenAI(
            base_url=PROXY_BASE_URL,
            api_key=os.environ.get("BRAINTRUST_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            timeout=45.0,
            max_retries=3,
        )
    )


def _build_prompt(behavior: BehaviorSection, context: str, transcript: str) -> str:
    return f"""You are grading whether a voice agent's call trajectory adhered to ONE behavior from an Agent Behavior spec.

Grade only this behavior. Do not reward or penalize conduct that belongs to a different behavior.

<behavior>
{behavior.body}
</behavior>

<call_context>
{context}
</call_context>

<trajectory>
{transcript}
</trajectory>

The trajectory is a transcript of a spoken phone call. Lines marked CALLER are what the
simulated caller said; lines marked AGENT are what the agent said, as transcribed from the
audio it actually produced. Bracketed lines are call events (interruptions, silence,
unintelligible audio).

First decide whether this behavior's triggering situation actually occurred in this
call. If it did not, the verdict is "na" -- even if the agent behaved well, and even
if the transcript shows conduct you approve of. Good conduct in a situation the
behavior does not cover is not evidence for that behavior.

Return exactly one verdict:
- "true": the situation this behavior describes occurred in the trajectory, and the agent exhibited the expected conduct.
- "false": the situation occurred, but the agent did not exhibit the expected conduct (including the failure modes the behavior warns against).
- "na": the situation this behavior describes did not occur in this trajectory, or the trajectory does not contain enough evidence to decide.

Respond with a JSON object only, in the form:
{{"verdict": "true" | "false" | "na", "rationale": "<one sentence citing evidence from the trajectory>"}}"""


def _parse_verdict(raw: str) -> BehaviorJudgment | None:
    """Parse the judge's reply, or None if it did not produce a usable verdict.

    Deliberately NOT falling back to `na`: that maps to a null score and is dropped
    from the average, so a judge that stopped returning verdicts would look exactly
    like a run where no behavior happened to apply. A parse failure is a broken
    judge and has to be distinguishable from a legitimate "did not apply".
    """
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    verdict = str(parsed.get("verdict", "")).lower()
    if verdict not in ("true", "false", "na"):
        return None
    rationale = parsed.get("rationale")
    return BehaviorJudgment(
        verdict=verdict,  # type: ignore[arg-type]
        rationale=rationale if isinstance(rationale, str) else "",
    )


async def judge_behavior(
    behavior: BehaviorSection, context: str, transcript: str
) -> BehaviorJudgment:
    """Grade one behavior over one call transcript.

    Asks for JSON mode so the reply is structurally guaranteed, retries once if the
    model still returns something unusable, and raises rather than inventing a
    verdict. A scorer that errors is visible in the experiment; a scorer that
    quietly returns `na` is not.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(behavior, context, transcript)},
    ]
    raw = ""
    for attempt in range(2):
        # Nothing provider-specific here on purpose. `response_format` is an
        # OpenAI-shaped parameter: through the gateway an Anthropic model accepts
        # it and returns an empty `{}`, which is worse than an error. The prompt
        # asks for JSON and `_parse_verdict` extracts it, which works on every
        # provider -- so swapping JUDGE_MODEL is all it takes to change model.
        response = await _client().chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=messages,  # type: ignore[arg-type]
        )
        raw = response.choices[0].message.content or ""
        judgment = _parse_verdict(raw)
        if judgment is not None:
            return judgment
        logger.warning(
            "judge returned an unusable verdict for %s (attempt %d): %r",
            behavior.slug,
            attempt + 1,
            raw[:200],
        )

    raise ValueError(
        f"judge did not return a usable verdict for {behavior.slug!r} "
        f"after 2 attempts; last reply: {raw[:200]!r}"
    )


def verdict_to_score(verdict: Verdict) -> float | None:
    """true -> 1, false -> 0, na -> None (excluded from the average)."""
    if verdict == "true":
        return 1.0
    if verdict == "false":
        return 0.0
    return None
