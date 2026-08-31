#!/usr/bin/env bash
# Aggregate LLM cost across EVERY project in the org.
#
#   ./scripts/org_costs.sh              # last 30 days, per project
#   ./scripts/org_costs.sh 7            # last 7 days
#   ./scripts/org_costs.sh 30 model     # break down by model instead
#
# Why a script and not one static query: `project_logs()` aggregates across
# multiple project IDs, but there is no wildcard for "all projects" -- the IDs
# have to be enumerated. `bt projects list` supplies them, and this splices them
# into a single query.
set -euo pipefail

DAYS="${1:-30}"
GROUP="${2:-project}"

case "$GROUP" in
  project) DIM="project_id" ;;
  model)   DIM="metadata.model" ;;
  day)     DIM="date_trunc('day', created)" ;;
  *) echo "usage: $0 [days] [project|model|day]" >&2; exit 2 ;;
esac

# Quoted, comma-separated project IDs.
IDS=$(bt projects list --json --no-color \
  | python3 -c "import json,sys;print(', '.join(f\"'{p['id']}'\" for p in json.load(sys.stdin)))")

COUNT=$(printf '%s' "$IDS" | tr -cd ',' | wc -c | tr -d ' ')
echo "querying $((COUNT + 1)) projects over the last ${DAYS}d, grouped by ${GROUP}..." >&2

# The `created` filter is not optional: without a range filter every query scans
# the project's full history and will time out on a large org.
# Project IDs are opaque in the output, so names are joined back in afterwards --
# Braintrust SQL has no JOIN, and there is no projects table to join against.
bt sql --json "
SELECT
  ${DIM}                              AS dimension,
  sum(estimated_cost())               AS total_cost,
  sum(metrics.tokens)                 AS tokens,
  sum(metrics.prompt_tokens)          AS prompt_tokens,
  sum(metrics.completion_tokens)      AS completion_tokens,
  count(1)                            AS spans
FROM project_logs(${IDS})
WHERE created > now() - interval ${DAYS} day
GROUP BY ${DIM}
HAVING sum(estimated_cost()) > 0
ORDER BY total_cost DESC
" --no-color | python3 -c "
import json, sys, subprocess

rows = json.loads(sys.stdin.read())
rows = rows.get('rows') or rows.get('data') or rows

label = {}
if '${GROUP}' == 'project':
    projects = json.loads(
        subprocess.run(['bt', 'projects', 'list', '--json', '--no-color'],
                       capture_output=True, text=True).stdout
    )
    label = {p['id']: p['name'] for p in projects}

total = 0.0
print(f\"{'name':<44}{'cost':>12}{'tokens':>12}{'spans':>9}\")
print('-' * 77)
for r in rows:
    dim = str(r.get('dimension') or '(none)')
    cost = float(r.get('total_cost') or 0)
    total += cost
    print(f\"{label.get(dim, dim)[:43]:<44}{cost:>12.4f}{int(r.get('tokens') or 0):>12,}{int(r.get('spans') or 0):>9,}\")
print('-' * 77)
print(f\"{'ORG TOTAL':<44}{total:>12.4f}\")
"
