"""Run one simulated call locally and print the transcript.

    uv run python -m voice_behaviors.simulation hours-probe

Useful for eyeballing a trajectory before spending an eval run on it.
"""

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from ..scenarios import SCENARIOS, by_id
from .caller import CallerPersona
from .runner import run_simulated_call


async def _main(scenario_id: str) -> None:
    scenario = by_id(scenario_id)
    persona = CallerPersona.from_metadata(scenario["metadata"])

    print(f"=== {scenario_id} ===")
    print(f"opening: {scenario['input']}\n")

    result = await run_simulated_call(scenario["input"], persona)

    print(result.transcript)
    print(f"\n--- ended: {result.ended_because} after {result.duration_s:.1f}s ---")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        default=SCENARIOS[0]["metadata"]["scenario_id"],
        help="scenario id from voice_behaviors.scenarios",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    asyncio.run(_main(args.scenario))


if __name__ == "__main__":
    main()
