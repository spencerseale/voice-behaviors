"""Push the simulated callers into a Braintrust dataset.

    uv run python scripts/seed_dataset.py

Each row is one caller: `input` is the opening phrase they say first, `metadata`
carries the persona that drives the rest of the conversation. Re-running upserts by
scenario id rather than duplicating rows.
"""

from __future__ import annotations

import sys

from braintrust import init_dataset
from dotenv import load_dotenv

from voice_behaviors.config import BRAINTRUST_PROJECT
from voice_behaviors.scenarios import SCENARIOS

DATASET_NAME = "voice-call-scenarios"


def main() -> int:
    load_dotenv()

    dataset = init_dataset(project=BRAINTRUST_PROJECT, name=DATASET_NAME)
    for scenario in SCENARIOS:
        dataset.insert(
            # Stable id keyed on the scenario, so re-seeding updates a row in
            # place instead of growing the dataset on every run.
            id=scenario["metadata"]["scenario_id"],
            input=scenario["input"],
            metadata=scenario["metadata"],
        )
    dataset.flush()

    summary = dataset.summarize()
    print(f"seeded {len(SCENARIOS)} scenarios into {BRAINTRUST_PROJECT}/{DATASET_NAME}")
    for scenario in SCENARIOS:
        meta = scenario["metadata"]
        targets = ", ".join(meta["targets"]) or "baseline"
        print(f"  {meta['scenario_id']:<14} -> {targets}")
    print(f"\n{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
