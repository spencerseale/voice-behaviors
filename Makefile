.PHONY: agent dev dataset sim eval test sync clean

CASE ?= hours-probe

# Interactive local session (mic/speaker). Ctrl+C to hang up.
agent: sync
	uv run voice-behaviors

# Connect to LiveKit Cloud and wait for dispatch instead of running locally.
dev: sync
	uv run voice-behaviors dev

# Push the simulated callers to the Braintrust dataset.
dataset: sync
	uv run python scripts/seed_dataset.py

# Run one simulated call locally and print the transcript: make sim CASE=interrupter
sim: sync
	uv run python -m voice_behaviors.simulation $(CASE)

# Grade every scenario against the behavior spec.
eval: sync
	uv run bt eval --language python evals/

# Offline unit tests -- no LiveKit, no OpenAI, no Braintrust. Fast.
test: sync
	uv run pytest -q tests/

sync:
	uv sync --quiet

clean:
	rm -rf .venv build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
