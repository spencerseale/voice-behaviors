"""Log one sample voice-call trace and validate saved Braintrust functions.

Usage:
    .venv/bin/python scripts/sample_trace.py
    .venv/bin/python scripts/sample_trace.py --scenario mumbler
    .venv/bin/python scripts/sample_trace.py --static

By default this runs the real in-process voice-to-voice simulator, logs the
result to project logs, and invokes the saved online scorers plus the custom
facet on the same call payload.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from typing import Any

from braintrust import Attachment, SpanTypeAttribute, init_logger
from dotenv import load_dotenv

from voice_behaviors.behaviors import load_behavior_spec
from voice_behaviors.config import BRAINTRUST_PROJECT, LLM_MODEL
from voice_behaviors.scenarios import SCENARIOS, by_id
from voice_behaviors.simulation import CallResult, CallTurn, CallerPersona
from voice_behaviors.simulation import run_simulated_call
from voice_behaviors.simulation.runner import call_context
from voice_behaviors.tracing import JOB_ENTRYPOINT_SPAN, log_call_spans

FACET_SLUG = "voice_call_friction"


def _static_result() -> CallResult:
    transcript = "\n".join(
        [
            "AGENT: Hi! Thanks for calling. How can I help you today?",
            "CALLER: Hi, what time do you close today?",
            (
                "AGENT: I do not have the business hours for Example Co. "
                "I can take a message or help route you to someone who would know."
            ),
            "CALLER: Okay, please have someone call me back.",
            "AGENT: Sure, I can take that message.",
            (
                "[call events: no interruption occurred at any point in this "
                "call. The two never spoke at the same time, turns strictly "
                "alternated, and no agent utterance was cut off. Neither party "
                "ever had to yield the floor.]"
            ),
        ]
    )
    return CallResult(
        transcript=transcript,
        turns=[
            CallTurn(
                "agent", "Hi! Thanks for calling. How can I help you today?"
            ),
            CallTurn("caller", "Hi, what time do you close today?"),
            CallTurn(
                "agent",
                (
                    "I do not have the business hours for Example Co. I can "
                    "take a message or help route you to someone who would know."
                ),
            ),
            CallTurn("caller", "Okay, please have someone call me back."),
            CallTurn("agent", "Sure, I can take that message."),
        ],
        ended_because="static sample",
        duration_s=42.0,
        usage={},
        agent_turns=[],
        spoken_turns=[],
    )


def _metadata(scenario: dict[str, Any], result: CallResult) -> dict[str, Any]:
    metadata = dict(scenario.get("metadata") or {})
    metadata.update(
        {
            "sample_trace": True,
            "entrypoint": "sample_trace",
            "ended_because": result.ended_because,
            "duration_s": round(result.duration_s, 1),
            "livekit_usage": result.usage,
            "agent_model": LLM_MODEL,
            "call_context": call_context(),
        }
    )
    if result.audio:
        metadata["call_audio"] = Attachment(
            data=result.audio,
            filename=f"{metadata.get('scenario_id', 'sample')}.wav",
            content_type="audio/wav",
        )
    return metadata


async def _run_and_log_trace(
    scenario_id: str, static: bool
) -> tuple[dict[str, Any], CallResult, dict[str, Any]]:
    scenario = by_id(scenario_id)
    logger = init_logger(project=BRAINTRUST_PROJECT)
    with logger.start_span(
        name=JOB_ENTRYPOINT_SPAN,
        type=SpanTypeAttribute.TASK,
        set_current=True,
    ) as root:
        if static:
            result = _static_result()
        else:
            persona = CallerPersona.from_metadata(scenario["metadata"])
            result = await run_simulated_call(scenario["input"], persona)

        metadata = _metadata(scenario, result)
        root.log(
            input=scenario["input"],
            output=result.as_output(),
            metadata=metadata,
        )
        if not result.trace_spans_logged:
            log_call_spans(result)

        trace_info = {
            "span_id": root.span_id,
            "root_span_id": root.root_span_id,
            "row_id": root.id,
            "permalink": root.permalink(),
        }
    logger.flush()
    return scenario, result, trace_info


def _validation_payload(scenario: dict[str, Any], result: CallResult) -> dict[str, Any]:
    metadata = dict(scenario.get("metadata") or {})
    metadata.update(
        {
            "sample_trace": True,
            "entrypoint": "sample_trace",
            "ended_because": result.ended_because,
            "duration_s": round(result.duration_s, 1),
            "agent_model": LLM_MODEL,
            "call_context": call_context(),
        }
    )
    return {
        "input": scenario["input"],
        "output": result.as_output(),
        "metadata": metadata,
        # The saved prompt scorers use this when Braintrust supplies a
        # trace-level preprocessed view; direct function calls need it
        # supplied explicitly.
        "preprocessed": result.transcript,
    }


def _invoke_function(slug: str, function_type: str, payload: dict[str, Any]) -> Any:
    result = subprocess.run(
        [
            "bt",
            "functions",
            "invoke",
            slug,
            "--type",
            function_type,
            "--project",
            BRAINTRUST_PROJECT,
            "--input",
            json.dumps(payload),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout)


def _print_scorer_result(slug: str, result: dict[str, Any]) -> None:
    metadata = result.get("metadata") or {}
    choice = metadata.get("choice") or metadata.get("label") or "?"
    score = result.get("score")
    rationale = metadata.get("rationale") or ""
    print(f"  [scorer] {slug}: score={score} choice={choice}")
    if rationale:
        print(f"           {rationale}")


def _print_facet_result(result: Any) -> None:
    output = result.get("output", result) if isinstance(result, dict) else result
    if isinstance(output, dict):
        output = output.get("value") or output.get("output") or output
    print(f"  [facet]  {FACET_SLUG}: {output}")


def validate_functions(payload: dict[str, Any]) -> None:
    spec = load_behavior_spec()
    print("\nSaved function validation:")
    for section in spec.sections:
        result = _invoke_function(section.slug, "scorer", payload)
        _print_scorer_result(section.slug, result)

    result = _invoke_function(FACET_SLUG, "facet", payload)
    _print_facet_result(result)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Log one sample Braintrust trace and validate saved functions."
    )
    parser.add_argument(
        "--scenario",
        default=SCENARIOS[0]["metadata"]["scenario_id"],
        help="Scenario ID from voice_behaviors.scenarios.",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use a deterministic text-only trace instead of running LiveKit audio.",
    )
    parser.add_argument(
        "--no-validate-functions",
        action="store_true",
        help="Only log the sample trace; do not invoke saved functions.",
    )
    args = parser.parse_args()

    if not os.environ.get("BRAINTRUST_API_KEY"):
        sys.exit("BRAINTRUST_API_KEY is not set.")

    scenario, result, trace_info = asyncio.run(
        _run_and_log_trace(args.scenario, args.static)
    )

    print(f"Logged sample trace for scenario={args.scenario}")
    print(f"  root_span_id={trace_info['root_span_id']}")
    print(f"  row_id={trace_info['row_id']}")
    print(f"  permalink={trace_info['permalink']}")

    if not args.no_validate_functions:
        validate_functions(_validation_payload(scenario, result))


if __name__ == "__main__":
    main()
