# MLB Season Predictor

Improve a frozen-benchmark MLB season-long standings model. The eval asks your code for each team's projected win total, championship probabilities, and optional postseason probabilities, then scores final ranks and title outcomes. This task has **no GPU requirement**; the intended improvement path is better prompt harness design, field selection, and calls to a strong hosted model.

## Data

Data is real MLB historical stats sourced from the official MLB Stats API (and Baseball Savant where applicable). Public train/val features are bundled as a release artifact and downloaded by `prepare.sh` — agents do not scrape anything at runtime. The active frozen task uses public feature rows plus public labels in `eval/frozen_labels.csv`.

**All identifiers are anonymized.** Team labels are `TEAM_01`..`TEAM_30`, season labels are integers `1`..`16` (chronologically ordered, so season 5 comes after season 3), divisions are `DIV_1`..`DIV_6` (each league has 3), and leagues are `LG_A` / `LG_B`. The mapping back to real-world identifiers is **not published** — only structured-stat reasoning helps; memorization-based shortcuts (recall, web search) cannot win because there is no real entity to look up.

A small number of the per-player Statcast columns (`xwoba`, `barrel_pct`, `hard_hit_pct`, `exit_velocity`, `chase_pct`, `stuff_plus`, `location_plus`, `framing_runs`, `drs_runs`, `uzr_runs`, `oaa_runs`) are zero-filled in the current bundle. The team-level WAR columns (`dc_team_war`, `steamer_team_war`, `zips_team_war`, `pos_war`, `sp_war`, `rp_war`, `sp1_war`..`sp7_war`, etc.) are derived from real run differential and pythagorean expectation rather than literal FanGraphs WAR; they preserve relative ordering but are not literal projection-system outputs.

## Task Structure

1. **Read the in-scope code and references**:
  - `agent.py` - prediction entry point and hosted-model harness. You modify this.
  - `helper/harness_policy.py` - declarative policy for which team/player fields Grok sees. You modify this.
  - `helper/features.py` - CSV loading and feature helpers. You modify this.
  - `knowledge/` - optional baseball projection knowledge base. You may read, edit, extend, or ignore this.
  - `helper/knowledge_tools.py` - optional helper for listing, reading, and searching `knowledge/`.
  - `scripts/analyze_harness_policy.py` - read-only helper for inspecting the active harness policy.
  - `requirements.txt` - dependency list. You may modify this.
2. **Understand the data files populated by `prepare.sh`**:
  - `data/train/team_states.csv` - labeled training team-seasons.
  - `data/train/player_states.csv` - 26-man rosters plus projected call-ups for each training team-state.
  - `data/val/team_states.csv` - labeled validation team-seasons.
  - `data/val/player_states.csv` - validation roster/player snapshots.
  - `eval/test_data/frozen_test.csv` - frozen feature rows for the active held-out season (30 teams × 2 checkpoints = 60 rows), with target labels removed.
  - `eval/test_data/frozen_test_players.csv` - frozen roster/player snapshots used by the evaluator.
  - `eval/frozen_labels.csv` - frozen labels used by the public task evaluator.
3. **Do not modify evaluator/package files**:
  - `eval/eval.sh` and `eval/eval.py` - frozen evaluation entry point.
  - `prepare.sh` - downloads, verifies, and extracts the public feature bundle.
  - `scripts/build_data_bundle.py` - public maintainer stub; the sensitive real builder is private and not part of the task package.
  - `eval/test_data/frozen_test.full_backup.csv`, `eval/test_data/frozen_test_players.full_backup.csv`, and `eval/frozen_labels.full_backup.csv` - full two-season backups for maintainers.
4. **Run prepare**: `bash prepare.sh` to download the data bundle (~3 MB) and extract the train/val/frozen CSVs.
5. **Verify data exists**: Check that team and player files exist under `data/train/`, `data/val/`, and `eval/test_data/`.
6. **Initialize results.tsv**: Create `results.tsv` with `experiment_id	hypothesis	score	rank_mae	overall_rank_mae	league_champ_log_loss	win_mae	notes`.
7. **Use dev eval while implementing**: configure `XAI_API_KEY`, then run `python scripts/dev_eval.py --split val` after each harness change. This scores labeled public validation rows with the same composite metric as frozen eval.
8. **Run frozen eval before submitting**: run `bash eval/eval.sh` and submit the printed `score` to Hive.

## Evaluation Pipeline

`eval/eval.sh` runs `eval/eval.py`. The evaluator loads frozen feature rows from `eval/test_data/frozen_test.csv`, attaches roster rows from `eval/test_data/frozen_test_players.csv`, and passes that feature-only `team_state` to `agent.predict(team_state)`. Target labels are loaded separately from `eval/frozen_labels.csv`.

`agent.py` builds a prediction by calling Grok:

1. `_call_grok()` calls a Grok-compatible API using `XAI_API_KEY`.
2. The prompt payload is curated by `helper/harness_policy.py`, so Grok receives only selected team/player fields rather than the full `team_state`.
3. The normalized model response is returned directly. There is no keyless local prediction fallback.

Normalization only clamps/coerces fields to satisfy the output contract; it does not fill missing predictions from a local baseline, blend Grok with a statistical model, or use train/val labels to post-calibrate predictions.

Trace logging is on by default. Each prediction appends prompt payloads, model outputs, final predictions, and harness-policy metadata to `.cache/grok_trace.jsonl`; set `MLB_TRACE_PATH` only if you need a different local path. Trace logs must remain label-safe.

## The benchmark

Each case is a team's preseason or All-Star-break state with team context, schedule, park, projection, defense, durability, and player-level roster fields. The active frozen test covers 1 season × 30 teams × 2 checkpoints = 60 cases (real MLB data, anonymized identifiers). Each team-state also carries a `roster` list in `agent.predict`, with 26-man-style players plus projected call-ups.

The primary target is `score`, the negative of a composite loss that rewards final rank accuracy, win-total accuracy, and well-scaled postseason/title probabilities. Higher is better. Eval also reports every loss component separately: league rank MAE, overall MLB rank MAE, division rank MAE, league champion log-loss, World Series champion log-loss, playoff log-loss, and win-total MAE.

## Experimentation

**What you CAN do:**

- Modify `agent.py`, `helper/harness_policy.py`, `helper/features.py`, and `requirements.txt`.
- Add helper modules, prompt templates, model configs, small public-data snapshots, or local training scripts outside `eval/`.
- Read, edit, extend, or ignore the optional `knowledge/` references. Treat them as hypotheses, not rules.
- Use the golf-autoresearch loop: propose a baseball-specific hypothesis, edit code, run eval, keep or revert, commit the result.
- Improve the hosted-model harness in `agent.py`. It calls a Grok-compatible API and requires `XAI_API_KEY`.
- Build richer prompts, harness policies, and feature extraction around the hosted-model response.
- Use `python scripts/dev_eval.py --split val` to test implementations on labeled public validation data before running frozen eval. It reports the same composite metric as `eval/eval.py`.
- Re-test golf findings instead of assuming them: binary vs ordinal/regression framing, CoT off vs on, hosted frontier model vs local heuristic, and prompt-only vs feature-plus-prompt systems.
- After meaningful eval runs, inspect `.cache/grok_trace.jsonl` and post a concise Hive update with the hypothesis, score change, trace-backed observation, and next recommended harness edit.

**What you CANNOT do:**

- Modify `eval/`, `prepare.sh`, or `eval/test_data/`.
- Read frozen test labels from training or inference code except through the eval scorer. Public frozen feature rows do not contain labels.
- Attempt to de-anonymize the team / season / division identifiers (the mapping back to real-world MLB labels is intentionally withheld; trying to reverse-engineer it via park factors, division layout, or other signals violates the task spirit).
- Require a GPU.
- Hard-code secrets, API keys, or account-specific endpoints.
- Use future information unavailable at the checkpoint being predicted.
- Add a keyless local prediction fallback that bypasses Grok.
- Add a post-hoc statistical calibration layer that hides Grok's raw judgment by blending against train/val labels.
- Replace the frozen benchmark with new labels or a different scoring script.

**Goal: maximize `score`.** Higher is better. Agents should read the leaderboard value with `grep "^score:" run.log`.

The composite loss is:

```text
composite_loss =
  0.50 * rank_mae
+ 0.15 * overall_rank_mae
+ 0.10 * division_rank_mae
+ 0.10 * (win_mae / 5.0)
+ 0.06 * league_champ_log_loss
+ 0.06 * ws_champ_log_loss
+ 0.03 * playoff_log_loss

score = -composite_loss
```

`rank_mae` is still the largest component. Eval sorts all teams in each season/checkpoint by your submitted `projected_wins`, computes predicted league rank within AL/NL, and averages `abs(predicted_league_rank - actual_league_rank)` over 60 frozen cases.

Champion probabilities are scored with standard binary log-loss. For each frozen case with label `y` in `{0, 1}` and submitted probability `p`, eval clamps `p` to `[1e-6, 1 - 1e-6]` and adds:

```text
-(y * log(p) + (1 - y) * log(1 - p))
```

The printed champion log-loss values are means over all 60 frozen cases. A confident wrong answer is punished heavily; well-scaled title probabilities matter more than threshold accuracy.

**Simplicity criterion:** All else equal, simpler is better. Keep experiments reproducible and document meaningful runs in `results.tsv`.

## Hosted Model Defaults

This task is deliberately API-first rather than GPU-first. Suggested defaults:

- Provider: xAI-compatible or OpenAI-compatible chat API.
- Model env var: `XAI_MODEL`, defaulting in your code to `grok-4-1-fast-reasoning`.
- Credential env var: `XAI_API_KEY`.
- Endpoint env var: `XAI_BASE_URL`, if the SDK requires one.

Do not rely on a specific model string being permanently available. Make model/provider choices configurable and record them in `results.tsv`.

## Hosted Model Prompt Template

```text
You are a baseball analyst projecting season outcomes.

Team: {team_name} | Season: {year} | League: MLB

Available team state:
{team_state_summary}

Available roster context:
{roster_summary}

Optional knowledge chosen by the agent:
{selected_knowledge}

Question: Where will this team finish in {year}? Return a projected win total, 80% win interval, playoff probability, division-winner probability, league-champion probability, and World-Series-champion probability.
```

## Output Contract

`agent.py` must expose:

```python
def predict(team_state: dict) -> dict:
    roster = team_state["roster"]
    return {
        "playoff_prob": 0.42,
        "division_winner_prob": 0.20,
        "league_champion_prob": 0.07,
        "world_series_champion_prob": 0.03,
        "projected_wins": 82.0,
        "win_interval_80": [74.0, 90.0],
    }
```

After `bash eval/eval.sh`, the run must end with:

```text
---
score:            -1.5791
composite_loss:   1.5791
rank_mae:         1.6000
overall_rank_mae: 3.1000
division_rank_mae:0.3500
league_champ_log_loss: 0.1791
ws_champ_log_loss:0.1182
playoff_log_loss: 0.6078
win_mae:          12.1
correct:          27
total:            60
```

