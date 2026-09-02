"""Create/update Braintrust online scoring rules for voice behavior scorers.

Usage:
    .venv/bin/python scripts/automations.py
    .venv/bin/python scripts/automations.py --list
    .venv/bin/python scripts/automations.py --sampling 0.25

The rules are trace-level/root-span rules because each behavior is judged over a
whole call trajectory, not over an individual STT, TTS, LLM, or turn span.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

from voice_behaviors.behaviors import load_behavior_spec
from voice_behaviors.config import BRAINTRUST_PROJECT

load_dotenv()

API_URL = os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev").rstrip("/")
PROJECT_NAME = os.environ.get("BRAINTRUST_PROJECT", BRAINTRUST_PROJECT)
DEFAULT_SAMPLING = float(os.environ.get("VOICE_BEHAVIOR_SAMPLING", "0.25"))


@dataclass(frozen=True)
class Rule:
    """One online scoring rule to reconcile."""

    name: str
    description: str
    scorer_slug: str
    sampling_rate: float


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
        timeout=60.0,
    )


def _objects(payload: Any) -> list[dict[str, Any]]:
    """Braintrust list endpoints return {"objects": [...]}."""
    if isinstance(payload, dict):
        return payload.get("objects", [])
    return payload if isinstance(payload, list) else []


def _validate_sampling(raw: str | float) -> float:
    value = float(raw)
    if not 0 < value <= 1:
        sys.exit(f"Sampling rate must be in (0, 1], got {value!r}.")
    return value


def resolve_project_id(client: httpx.Client, name: str) -> str:
    response = client.get("/v1/project", params={"project_name": name})
    response.raise_for_status()
    projects = _objects(response.json())
    if not projects:
        sys.exit(
            f"Project {name!r} not found. Run the agent once or set "
            "BRAINTRUST_PROJECT."
        )
    return projects[0]["id"]


def resolve_scorer_ids(client: httpx.Client, project_id: str) -> dict[str, str]:
    """Map scorer slug to function ID for every scorer in the project."""
    response = client.get(
        "/v1/function",
        params={
            "project_id": project_id,
            "function_type": "scorer",
            "limit": 200,
        },
    )
    response.raise_for_status()
    return {f["slug"]: f["id"] for f in _objects(response.json()) if f.get("slug")}


def list_rules(client: httpx.Client, project_id: str) -> list[dict[str, Any]]:
    response = client.get(
        "/v1/project_score", params={"project_id": project_id, "limit": 200}
    )
    response.raise_for_status()
    return [s for s in _objects(response.json()) if s.get("score_type") == "online"]


def behavior_rules(sampling_rate: float) -> list[Rule]:
    spec = load_behavior_spec()
    return [
        Rule(
            name=f"{section.title} (voice conduct)",
            description=(
                "Trace-level online scoring rule for the voice-call conduct "
                f"behavior {section.title!r}. Non-applicable traces are skipped "
                "by the prompt scorer."
            ),
            scorer_slug=section.slug,
            sampling_rate=sampling_rate,
        )
        for section in spec.sections
    ]


def upsert_rule(
    client: httpx.Client, project_id: str, rule: Rule, scorer_id: str
) -> dict[str, Any]:
    """Create or replace one online scoring rule by name."""
    body = {
        "project_id": project_id,
        "name": rule.name,
        "description": rule.description,
        "score_type": "online",
        "config": {
            "online": {
                "sampling_rate": rule.sampling_rate,
                "scorers": [{"type": "function", "id": scorer_id}],
                "apply_to_root_span": True,
            }
        },
    }
    response = client.put("/v1/project_score", json=body)
    if response.status_code >= 400:
        sys.exit(
            f"Failed to upsert {rule.name!r}: "
            f"{response.status_code} {response.text}"
        )
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure Braintrust online scoring rules."
    )
    parser.add_argument("--list", action="store_true", help="List existing rules.")
    parser.add_argument(
        "--sampling",
        type=_validate_sampling,
        default=DEFAULT_SAMPLING,
        help=(
            "Sampling rate for each behavior rule, as a 0-1 fraction. "
            "Default: %(default)s."
        ),
    )
    args = parser.parse_args()

    with _client() as client:
        project_id = resolve_project_id(client, PROJECT_NAME)
        print(f"Project: {PROJECT_NAME} ({project_id})")

        if args.list:
            rules = list_rules(client, project_id)
            if not rules:
                print("No online scoring rules configured.")
                return
            print(f"\n{len(rules)} online scoring rule(s):")
            for rule in rules:
                online = (rule.get("config") or {}).get("online") or {}
                scorer_ids = ", ".join(
                    s.get("id", "?") for s in online.get("scorers", [])
                )
                print(
                    f"- {rule['name']}\n"
                    f"  sampling={online.get('sampling_rate')} "
                    f"root_span={online.get('apply_to_root_span')}\n"
                    f"  scorers=[{scorer_ids}]"
                )
            return

        rules = behavior_rules(args.sampling)
        available = resolve_scorer_ids(client, project_id)
        missing = [r.scorer_slug for r in rules if r.scorer_slug not in available]
        if missing:
            sys.exit(
                "Missing scorer(s): "
                f"{', '.join(missing)}. Run `make push-scorers` first."
            )

        print()
        for rule in rules:
            result = upsert_rule(
                client, project_id, rule, available[rule.scorer_slug]
            )
            print(
                f"[ok] {rule.name}\n"
                f"     scorer={rule.scorer_slug} "
                f"sampling={rule.sampling_rate:.0%} rule_id={result.get('id')}"
            )

        app_url = os.environ.get(
            "BRAINTRUST_APP_URL", "https://www.braintrust.dev"
        ).rstrip("/")
        print(
            "\nOnline behavior scoring rules are configured for new root traces.\n"
            f"Review at {app_url}/app/~/configuration/automations"
        )


if __name__ == "__main__":
    main()
