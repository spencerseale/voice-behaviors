.PHONY: agent dev dataset sim eval push-scorers automations topics topics-status online-status sample-trace test sync clean

CASE ?= hours-probe
PROJECT ?= voice-behaviors
PYTHON := $(CURDIR)/.venv/bin/python
SCORER_SAMPLING ?= 0.25

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

# Push the four behavior scorers to Braintrust (audio empathy intentionally omitted).
push-scorers: sync
	bt functions push scorers/voice_call_conduct.py \
		--project $(PROJECT) \
		--runner $(PYTHON) \
		--if-exists replace \
		--yes

# Create/update online scoring rules for the behavior scorers.
automations: sync
	$(PYTHON) scripts/automations.py --sampling $(SCORER_SAMPLING)

# Enable Topics and attach the custom voice_call_friction facet.
topics: sync
	$(PYTHON) scripts/topics.py

# Show the Topics automation config and status.
topics-status: sync
	$(PYTHON) scripts/topics.py --status
	bt topics status --project $(PROJECT) || true

# Show pushed scorers and online scoring rules.
online-status: sync
	bt scorers list --project $(PROJECT) || true
	$(PYTHON) scripts/automations.py --list

# Log one sample call trace and invoke the saved scorers/facet against it.
sample-trace: sync
	$(PYTHON) scripts/sample_trace.py --scenario $(CASE)

# Offline unit tests -- no LiveKit, no OpenAI, no Braintrust. Fast.
test: sync
	uv run pytest -q tests/

sync:
	uv sync --quiet

clean:
	rm -rf .venv build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
