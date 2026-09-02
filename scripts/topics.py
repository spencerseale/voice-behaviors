"""Enable Braintrust Topics and add the custom voice-call facet.

Usage:
    .venv/bin/python scripts/topics.py
    .venv/bin/python scripts/topics.py --status

The custom facet is open-ended. It summarizes the caller's need and any
user-visible conversational friction, then Braintrust clusters those summaries
into topics through a paired topic-map classifier.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

from voice_behaviors.config import BRAINTRUST_PROJECT
from voice_behaviors.simulation.runner import call_context

load_dotenv()

API_URL = os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev").rstrip("/")
PROJECT_NAME = os.environ.get("BRAINTRUST_PROJECT", BRAINTRUST_PROJECT)
FACET_MODEL = os.environ.get("VOICE_FACET_MODEL", "gpt-4o-mini")
TOPIC_WINDOW = os.environ.get("VOICE_TOPICS_WINDOW", "1d")
GENERATION_CADENCE = os.environ.get("VOICE_TOPICS_CADENCE", "1h")
TOPICS_SAMPLING_RATE = float(os.environ.get("VOICE_TOPICS_SAMPLING", "0.25"))

CUSTOM_FACET = {
    "name": "voice_call_friction",
    "description": (
        "Open-ended summary of the caller's need and any user-visible voice-call "
        "friction, for discovery-oriented topic clustering."
    ),
    "prompt": """You are analyzing a trace from an inbound phone-call voice agent for a demo business.

Identify the single most useful topic phrase for clustering this call. Prefer the
user-visible problem or friction if one appears; otherwise summarize the caller's
main need.

For this demo, the agent had only this call context:
{call_context}

Use that context when identifying unsupported business facts. If the caller asks
for a verifiable business fact and the agent answers with a specific fact absent
from this context and absent from the caller's own words, treat that as the
primary topic.

Look for:
- unsupported business facts, such as hours, prices, policies, services, products, locations, or account details the agent did not know
- garbled, empty, or ambiguous caller turns
- interruptions, talk-over, cut-off agent audio, or ignored barge-ins
- output that would be awkward when heard aloud, such as markdown, bullets, URLs, or long identifiers
- routing, transfer, callback, or message-taking handoffs
- routine caller intents when no failure or friction is visible

Output format: a lowercase noun phrase of 3 to 8 words. Do not use punctuation,
quotes, full sentences, labels, or explanations.

Good outputs:
unsupported business hours question
unsupported business hours claim
garbled caller request clarification
caller interruption and turn taking
routine message taking handoff
spoken markdown or url problem
business service availability question

If there is no readable conversation content in the trace, output exactly:
NONE""".format(call_context=call_context()),
}

NO_MATCH_PATTERN = "^NONE$"
TOPIC_MAP_SUFFIX = "_topic_map"


def _client() -> httpx.Client:
    api_key = os.environ.get("BRAINTRUST_API_KEY", "").strip()
    if not api_key:
        sys.exit("BRAINTRUST_API_KEY is not set.")
    return httpx.Client(
        base_url=API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=90.0,
    )


def _objects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return payload.get("objects", [])
    return payload if isinstance(payload, list) else []


def _bt(args: list[str], check: bool = True) -> str:
    result = subprocess.run(
        ["bt", *args],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if check and result.returncode != 0:
        sys.exit(f"`bt {' '.join(args)}` failed:\n{result.stderr or result.stdout}")
    return result.stdout


def _validate_sampling(value: float) -> float:
    if not 0 < value <= 1:
        sys.exit(f"VOICE_TOPICS_SAMPLING must be in (0, 1], got {value!r}.")
    return value


def resolve_project_id(client: httpx.Client, name: str) -> str:
    response = client.get("/v1/project", params={"project_name": name})
    response.raise_for_status()
    projects = _objects(response.json())
    if not projects:
        sys.exit(f"Project {name!r} not found. Run the agent once so it gets created.")
    return projects[0]["id"]


def topics_status() -> dict[str, Any]:
    raw = _bt(["topics", "config", "--json", "--project", PROJECT_NAME, "--no-input"])
    return json.loads(raw)


def ensure_topics_enabled() -> None:
    status = topics_status()
    if status.get("automations"):
        print("  [ok] Topics automation already exists")
        return

    _validate_sampling(TOPICS_SAMPLING_RATE)
    percent = f"{TOPICS_SAMPLING_RATE * 100:g}%"
    _bt(
        [
            "topics",
            "config",
            "enable",
            "--name",
            "Voice Behavior Topics",
            "--description",
            "Topics over voice-call traces and the custom voice_call_friction facet.",
            "--topic-window",
            TOPIC_WINDOW,
            "--generation-cadence",
            GENERATION_CADENCE,
            "--sampling-rate",
            percent,
            "--project",
            PROJECT_NAME,
            "--no-input",
        ]
    )
    print(
        "  [ok] Topics enabled "
        f"(built-in facets, window={TOPIC_WINDOW}, cadence={GENERATION_CADENCE}, "
        f"sampling={TOPICS_SAMPLING_RATE:.0%})"
    )


def create_facet_functions(client: httpx.Client, project_id: str) -> tuple[str, str]:
    name = CUSTOM_FACET["name"]
    result = client.post(
        "/v1/function",
        json={
            "project_id": project_id,
            "name": name,
            "slug": name,
            "description": CUSTOM_FACET["description"],
            "function_type": "facet",
            "if_exists": "replace",
            "function_data": {
                "type": "facet",
                "preprocessor": {
                    "type": "global",
                    "name": "thread",
                    "function_type": "preprocessor",
                },
                "prompt": CUSTOM_FACET["prompt"],
                "model": FACET_MODEL,
                "no_match_pattern": NO_MATCH_PATTERN,
            },
        },
    )
    if result.status_code >= 400:
        sys.exit(f"Facet creation failed: {result.status_code} {result.text}")
    facet_id = result.json()["id"]

    topic_map_name = f"{name}{TOPIC_MAP_SUFFIX}"
    topic_map = client.post(
        "/insert-functions",
        json={
            "functions": [
                {
                    "project_id": project_id,
                    "name": topic_map_name,
                    "slug": topic_map_name,
                    "function_type": "classifier",
                    "if_exists": "replace",
                    "function_data": {
                        "type": "topic_map",
                        "source_facet": name,
                        "embedding_model": "brain-embedding-1",
                    },
                }
            ]
        },
    )
    if topic_map.status_code >= 400:
        sys.exit(
            f"Topic map creation failed: {topic_map.status_code} {topic_map.text}"
        )
    functions = topic_map.json().get("functions", [])
    if not functions:
        sys.exit("Topic map creation returned no function ID.")
    topic_map_id = functions[0]["id"]

    print(
        f"  [ok] {name}: facet ({facet_id[:8]}...) "
        f"+ topic map ({topic_map_id[:8]}...)"
    )
    return facet_id, topic_map_id


def _facet_ref(entry: dict[str, Any]) -> dict[str, Any] | None:
    ref_type = entry.get("ref_type") or entry.get("type")
    if ref_type == "global":
        name = entry.get("name")
        if not name:
            return None
        return {"type": "global", "name": name, "function_type": "facet"}
    if ref_type == "function":
        fn_id = entry.get("id")
        if not fn_id:
            return None
        return {"type": "function", "id": fn_id}
    return None


def _topic_map_ref(entry: dict[str, Any]) -> dict[str, Any] | None:
    fn_id = entry.get("id")
    ref_type = entry.get("ref_type") or entry.get("type")
    function_type = entry.get("function_type")
    if not fn_id or function_type != "classifier":
        return None
    return {"function": {"type": ref_type or "function", "id": fn_id}}


def patch_automation(
    client: httpx.Client, facet_id: str, topic_map_id: str
) -> dict[str, Any]:
    status = topics_status()
    automations = status.get("automations") or []
    if not automations:
        sys.exit("Topics automation not found after enable step.")
    auto = automations[0]

    facet_functions = []
    seen_facet_refs: set[tuple[str, str]] = set()
    for entry in auto.get("facet_functions", []):
        ref = _facet_ref(entry)
        if not ref:
            continue
        key = (ref["type"], ref.get("id") or ref.get("name", ""))
        if key in seen_facet_refs or key == ("function", facet_id):
            continue
        seen_facet_refs.add(key)
        facet_functions.append(ref)
    facet_functions.append({"type": "function", "id": facet_id})

    topic_map_functions = []
    seen_topic_maps: set[str] = set()
    for entry in auto.get("topic_map_functions", []):
        ref = _topic_map_ref(entry)
        if not ref:
            continue
        fn_id = ref["function"]["id"]
        if fn_id in seen_topic_maps or fn_id == topic_map_id:
            continue
        seen_topic_maps.add(fn_id)
        topic_map_functions.append(ref)
    topic_map_functions.append({"function": {"type": "function", "id": topic_map_id}})

    scope_type = auto.get("scope_type") or (auto.get("scope") or {}).get("type") or "logs"
    response = client.post(
        "/api/project_automation/patch_id",
        json={
            "id": auto["id"],
            "config": {
                "event_type": "topic",
                "sampling_rate": auto.get("sampling_rate", TOPICS_SAMPLING_RATE),
                "facet_functions": facet_functions,
                "topic_map_functions": topic_map_functions,
                "scope": {
                    "type": scope_type,
                    "idle_seconds": auto.get("idle_seconds", 600),
                },
                "rerun_seconds": auto.get("rerun_seconds"),
                "relabel_overlap_seconds": auto.get("relabel_overlap_seconds"),
                "backfill_time_range": TOPIC_WINDOW,
            },
        },
    )
    if response.status_code >= 400:
        sys.exit(f"Automation patch failed: {response.status_code} {response.text}")
    print("  [ok] Topics automation patched with custom facet and topic map")
    return response.json()


def print_status() -> None:
    status = topics_status()
    automations = status.get("automations") or []
    if not automations:
        print("Topics is not configured for this project.")
        return

    for auto in automations:
        print(f"Topics automation: {auto.get('name', auto.get('id'))}")
        print(
            "  "
            f"scope={auto.get('scope_type')} "
            f"rerun_seconds={auto.get('rerun_seconds')} "
            f"sampling={auto.get('sampling_rate')}"
        )
        print("  facet functions:")
        for entry in auto.get("facet_functions", []):
            print(f"  - {entry.get('name') or entry.get('id')} ({entry.get('ref_type')})")
        print("  topic map functions:")
        for entry in auto.get("topic_map_functions", []):
            print(
                f"  - {entry.get('name') or entry.get('id')} "
                f"({entry.get('function_type')})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enable Topics and push the custom voice-call facet."
    )
    parser.add_argument("--status", action="store_true", help="Show Topics config.")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    with _client() as client:
        project_id = resolve_project_id(client, PROJECT_NAME)
        print(f"Project: {PROJECT_NAME} ({project_id})\n")

        ensure_topics_enabled()
        facet_id, topic_map_id = create_facet_functions(client, project_id)
        patch_automation(client, facet_id, topic_map_id)

    app_url = os.environ.get("BRAINTRUST_APP_URL", "https://www.braintrust.dev").rstrip("/")
    print(
        f"\nTopics is configured with custom facet: {CUSTOM_FACET['name']}.\n"
        f"Review at {app_url}/app/~/{PROJECT_NAME}/topics"
    )


if __name__ == "__main__":
    main()
