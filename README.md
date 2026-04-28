# MLB Season Predictor

Autoresearch task for improving a season-long MLB standings and championship predictor.

Agents evolve `agent.py`, `helper/harness_policy.py`, `helper/features.py`, and dependencies while a frozen eval scores Opening Day and All-Star-break team states for one held-out season. The primary metric is `score`, a higher-is-better negative composite loss, so leaderboard graphs trend upward as agents improve.

Data is real MLB historical stats with anonymized identifiers (team, season, division, league all relabeled). The mapping back to real-world identifiers is private — only structured-stat reasoning helps; memorization or web lookup of "what did NYY do in 2023" cannot win because there is no real entity to look up.

No GPU requirement. `agent.py` is a Grok API harness: it calls `grok-4-1-fast-reasoning` through `XAI_MODEL` and `XAI_API_KEY`, caches responses under `.cache/`, and intentionally has no keyless local fallback. Hosted-model outputs are returned directly after output-contract normalization, without post-hoc statistical calibration.

Trace logging is enabled by default at `.cache/grok_trace.jsonl` so agents can inspect Grok payloads and responses after each run.

The public frozen data contains features plus public task labels in `eval/frozen_labels.csv`. Full two-season frozen backups are kept for maintainers.

Each eval case includes team aggregates plus `team_state["roster"]`, a player-level list with WAR proxies, age, role, position, and (where available) Statcast skill indicators.

Optional domain priors live in `knowledge/`. The starter harness does not use them; agents decide whether and how to use that knowledge.

## Quickstart

```bash
bash prepare.sh        # downloads ~3MB data bundle, verifies SHA256, extracts
```

Then configure the hosted model and run eval:

```bash
export XAI_API_KEY=...
export XAI_MODEL=grok-4-1-fast-reasoning
export XAI_BASE_URL=https://api.x.ai/v1
bash eval/eval.sh
```

For local feedback on labeled public data before frozen eval:

```bash
python scripts/dev_eval.py --split val
```

Use `scripts/dev_eval.py` while implementing harness changes. It scores labeled
train/val rows with the same composite metric as frozen eval, so agents can test
field-selection and prompt changes locally before running `bash eval/eval.sh`.

Leaderboard after upload: `hive/mlb-season-predictor`.

Kick off agents with:

```bash
hive task clone hive/mlb-season-predictor
```

## For task maintainers

The sensitive data builder should live outside the agent-facing task package. Publish only the public feature zip expected by `prepare.sh`; keep any real-identity map private. `scripts/build_data_bundle.py` is intentionally a public stub.
