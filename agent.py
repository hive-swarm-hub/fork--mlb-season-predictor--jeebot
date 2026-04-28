"""Grok harness for the MLB Season Predictor hive task.

This file calls a Grok-compatible chat API and caches strict JSON predictions.
It intentionally has no local fallback: the task is to improve the hosted-model
harness and observe Grok's judgment directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from helper.features import clamp, load_rosters, load_rows, roster_key
from helper.harness_policy import FORBIDDEN_TARGET_KEYS, get_harness_policy, selected_player_keys, selected_team_keys
from helper.league_context import peer_summary


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache" / "grok_predictions"
BATCH_CACHE_DIR = ROOT / ".cache" / "grok_batch"
DEFAULT_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_TEMPERATURE = 0.2

# Disk locations for full team-state / roster lookup in batch mode.
_BATCH_CSV_CANDIDATES: tuple[tuple[Path, Path], ...] = (
    (ROOT / "eval" / "test_data" / "frozen_test.csv", ROOT / "eval" / "test_data" / "frozen_test_players.csv"),
    (ROOT / "data" / "val" / "team_states.csv", ROOT / "data" / "val" / "player_states.csv"),
    (ROOT / "data" / "train" / "team_states.csv", ROOT / "data" / "train" / "player_states.csv"),
)
import threading
_BATCH_LOCKS: dict[tuple[int, str, str], threading.Lock] = {}
_BATCH_LOCKS_GUARD = threading.Lock()


def _harness_fingerprint() -> str:
    """Bust Grok response cache when harness field selection changes."""
    pol = get_harness_policy()
    blob = json.dumps(
        {
            "player_keys": pol.get("player_keys", []),
            "selected_groups": pol.get("selected_groups", []),
            "source": pol.get("source", ""),
            "team_keys": pol.get("team_keys", []),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _cache_key(team_state: dict, model: str) -> str:
    payload = {
        "harness": _harness_fingerprint(),
        "model": model,
        "prompt_payload": _prompt_payload(team_state),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _top_players(roster: list[dict], role: str | None = None, limit: int = 8) -> list[dict]:
    rows = [p for p in roster if role is None or p.get("role") == role]
    return sorted(rows, key=lambda p: float(p.get("projection_blend_war", 0.0)), reverse=True)[:limit]


def _summarize_player(player: dict) -> dict:
    keys = selected_player_keys()
    return {key: player.get(key) for key in keys if key in player}


def _prompt_payload(team_state: dict) -> dict:
    """Return the exact label-safe payload sent to the model.

    Eval rows include target columns such as actual wins and final ranks. Keep
    prompt construction and tracing on this allowlist so diagnostics cannot
    accidentally become a frozen-label leak.
    """
    roster = team_state.get("roster", [])
    hitters = [_summarize_player(p) for p in _top_players(roster, None, 10) if p.get("role") in {"position_player", "bench", "callup"}]
    starters = [_summarize_player(p) for p in _top_players(roster, "starter", 7)]
    relievers = [_summarize_player(p) for p in _top_players(roster, "reliever", 5)]
    team_keys = selected_team_keys()
    team_summary = {key: team_state.get(key) for key in team_keys if key in team_state}
    peers = peer_summary(
        season=team_state.get("season"),
        checkpoint=team_state.get("checkpoint"),
        league=team_state.get("league"),
    )
    return {
        "task": "Project MLB final standings outcomes from this checkpoint. Use league_peers to calibrate where this team ranks within its league.",
        "team_state": team_summary,
        "league_peers": peers,
        "top_hitters": hitters,
        "starting_pitchers": starters,
        "high_leverage_relievers": relievers,
        "output_schema": {
            "projected_wins": "number",
            "playoff_prob": "0..1",
            "division_winner_prob": "0..1",
            "league_champion_prob": "0..1",
            "world_series_champion_prob": "0..1",
            "win_interval_80": ["low", "high"],
        },
    }


def _prompt(team_state: dict) -> str:
    payload = _prompt_payload(team_state)
    return (
        "Return only compact JSON matching output_schema. Do not include markdown.\n"
        + json.dumps(payload, sort_keys=True)
    )


def _trace_path() -> Path | None:
    configured = os.getenv("MLB_TRACE_PATH")
    path = Path(configured) if configured else ROOT / ".cache" / "grok_trace.jsonl"
    if not path.is_absolute():
        path = ROOT / path
    return path


def _trace_raw_enabled() -> bool:
    return os.getenv("MLB_TRACE_RAW", "1").lower() not in {"0", "false", "no"}


def _write_trace(event: dict) -> None:
    path = _trace_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _required_float(raw: dict, key: str) -> float:
    if key not in raw:
        raise ValueError(f"model response missing required field: {key}")
    return float(raw[key])


def _normalize_prediction(raw: dict) -> dict:
    projected_wins = clamp(_required_float(raw, "projected_wins"), 40.0, 122.0)
    interval = raw.get("win_interval_80")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("model response missing required field: win_interval_80")
    low = clamp(float(interval[0]), 35.0, projected_wins)
    high = clamp(float(interval[1]), projected_wins, 125.0)
    return {
        "playoff_prob": clamp(_required_float(raw, "playoff_prob"), 0.001, 0.999),
        "division_winner_prob": clamp(_required_float(raw, "division_winner_prob"), 0.001, 0.999),
        "league_champion_prob": clamp(_required_float(raw, "league_champion_prob"), 0.001, 0.999),
        "world_series_champion_prob": clamp(_required_float(raw, "world_series_champion_prob"), 0.001, 0.999),
        "projected_wins": projected_wins,
        "win_interval_80": [low, high],
    }


def _call_grok(team_state: dict, trace: dict | None = None) -> dict:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is required; this task has no local fallback")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required to call the Grok-compatible API") from exc

    model = os.getenv("XAI_MODEL", DEFAULT_MODEL)
    key = _cache_key(team_state, model)
    cache_path = CACHE_DIR / f"{key}.json"
    if trace is not None:
        trace["model"] = model
        trace["cache_key"] = key
        trace["prompt_payload"] = _prompt_payload(team_state)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        prediction = _normalize_prediction(cached)
        if trace is not None:
            trace["source"] = "grok_cache"
            trace["model_prediction"] = prediction
        return prediction

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
    )
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an MLB season forecasting model. "
                        "Use only the supplied team and roster state. Return strict JSON."
                    ),
                },
                {"role": "user", "content": _prompt(team_state)},
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Grok API request failed: {type(exc).__name__}: {exc}") from exc
    content = response.choices[0].message.content or "{}"
    elapsed_ms = round((time.time() - started) * 1000)
    raw = _extract_json(content)
    prediction = _normalize_prediction(raw)
    if trace is not None:
        trace["source"] = "grok_api"
        trace["elapsed_ms"] = elapsed_ms
        trace["raw_response_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        if _trace_raw_enabled():
            trace["raw_model_text"] = content
        trace["parsed_model_json"] = raw
        trace["model_prediction"] = prediction
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prediction, sort_keys=True) + "\n")
    return prediction

def _team_state_for_batch(row: dict, rosters: dict) -> dict:
    """Strip target labels and attach roster — label-safe view of a team-season."""
    safe = {k: v for k, v in row.items() if k not in FORBIDDEN_TARGET_KEYS}
    safe["roster"] = rosters.get(roster_key(row), [])
    return safe


def _league_team_states(season: int, checkpoint: str, league: str) -> list[dict]:
    """Load full label-safe team_states (with rosters) for every team in a league/checkpoint."""
    for team_csv, player_csv in _BATCH_CSV_CANDIDATES:
        if not team_csv.exists():
            continue
        rows = [
            r
            for r in load_rows(team_csv)
            if int(r["season"]) == int(season)
            and str(r["checkpoint"]) == str(checkpoint)
            and str(r["league"]) == str(league)
        ]
        if not rows:
            continue
        rosters = load_rosters(player_csv) if player_csv.exists() else {}
        return [_team_state_for_batch(r, rosters) for r in rows]
    return []


def _team_block(state: dict) -> dict:
    """Per-team payload block used inside the batch prompt."""
    roster = state.get("roster", [])
    hitters = [_summarize_player(p) for p in _top_players(roster, None, 10) if p.get("role") in {"position_player", "bench", "callup"}]
    starters = [_summarize_player(p) for p in _top_players(roster, "starter", 7)]
    relievers = [_summarize_player(p) for p in _top_players(roster, "reliever", 5)]
    team_keys = selected_team_keys()
    team_summary = {key: state.get(key) for key in team_keys if key in state}
    return {
        "team_id": state.get("team_id"),
        "division": state.get("division"),
        "team_state": team_summary,
        "top_hitters": hitters,
        "starting_pitchers": starters,
        "high_leverage_relievers": relievers,
    }


def _batch_payload(states: list[dict]) -> dict:
    return {
        "task": (
            "Project MLB final standings outcomes for ALL teams listed in this league at this checkpoint. "
            "Order the teams by expected wins so the rankings within and across divisions are internally consistent. "
            "Return a JSON object whose top-level keys are team_ids and whose values match output_schema. "
            "Cover every team in `teams`."
        ),
        "season": states[0].get("season") if states else None,
        "checkpoint": states[0].get("checkpoint") if states else None,
        "league": states[0].get("league") if states else None,
        "teams": [_team_block(s) for s in states],
        "output_schema": {
            "<team_id>": {
                "projected_wins": "number",
                "playoff_prob": "0..1",
                "division_winner_prob": "0..1",
                "league_champion_prob": "0..1",
                "world_series_champion_prob": "0..1",
                "win_interval_80": ["low", "high"],
            }
        },
    }


def _batch_cache_path(season: int, checkpoint: str, league: str, model: str, temperature: float) -> Path:
    payload = {
        "harness": _harness_fingerprint(),
        "model": model,
        "temperature": temperature,
        "season": int(season),
        "checkpoint": str(checkpoint),
        "league": str(league),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return BATCH_CACHE_DIR / f"{hashlib.sha256(blob.encode()).hexdigest()}.json"


def _batch_lock(key: tuple[int, str, str]) -> threading.Lock:
    with _BATCH_LOCKS_GUARD:
        lock = _BATCH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _BATCH_LOCKS[key] = lock
        return lock


def _ensure_batch_predictions(season: int, checkpoint: str, league: str) -> dict[str, dict]:
    """Return {team_id: prediction} for every team in (season, checkpoint, league).

    Uses a single Grok call per league/checkpoint and caches on disk so the
    eval can replay results across runs without re-spending tokens.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is required; this task has no local fallback")
    model = os.getenv("XAI_MODEL", DEFAULT_MODEL)
    temperature = float(os.getenv("XAI_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    cache_path = _batch_cache_path(season, checkpoint, league, model, temperature)
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    lock = _batch_lock((int(season), str(checkpoint), str(league)))
    with lock:
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        states = _league_team_states(season, checkpoint, league)
        if not states:
            raise RuntimeError(f"no team-states found for batch ({season}, {checkpoint}, {league})")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required to call the Grok-compatible API") from exc
        client = OpenAI(api_key=api_key, base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"))
        payload = _batch_payload(states)
        prompt_text = (
            "Return only compact JSON. Top-level keys are the team_ids listed in `teams`; values follow output_schema. "
            "Do not include markdown.\n"
            + json.dumps(payload, sort_keys=True)
        )
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an MLB season forecasting model. "
                        "Use only the supplied team and roster state. Return strict JSON."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
        )
        content = response.choices[0].message.content or "{}"
        raw = _extract_json(content)
        normalized: dict[str, dict] = {}
        for team_id, pred in raw.items():
            if not isinstance(pred, dict):
                continue
            try:
                normalized[str(team_id)] = _normalize_prediction(pred)
            except Exception:
                continue
        BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(normalized, sort_keys=True) + "\n")
        return normalized


def _batch_predict(team_state: dict) -> dict | None:
    season = team_state.get("season")
    checkpoint = team_state.get("checkpoint")
    league = team_state.get("league")
    team_id = team_state.get("team_id")
    if season is None or checkpoint is None or league is None or team_id is None:
        return None
    batch = _ensure_batch_predictions(int(season), str(checkpoint), str(league))
    return batch.get(str(team_id))


def predict(team_state: dict) -> dict:
    trace: dict | None = None
    if _trace_path() is not None:
        pol = get_harness_policy()
        trace = {
            "trace_version": 1,
            "case": {
                "season": team_state.get("season"),
                "checkpoint": team_state.get("checkpoint"),
                "team_id": team_state.get("team_id"),
                "league": team_state.get("league"),
                "division": team_state.get("division"),
            },
            "harness_policy": {
                "fingerprint": _harness_fingerprint(),
                "player_key_count": len(selected_player_keys()),
                "selected_groups": pol.get("selected_groups", []),
                "source": pol.get("source", ""),
                "team_key_count": len(selected_team_keys()),
            },
        }

    use_batch = os.getenv("MLB_BATCH_MODE", "1").lower() not in {"0", "false", "no"}
    prediction: dict | None = None
    if use_batch:
        try:
            prediction = _batch_predict(team_state)
            if prediction is not None and trace is not None:
                trace["source"] = "grok_batch"
        except Exception as exc:
            if trace is not None:
                trace["batch_error"] = f"{type(exc).__name__}: {exc}"
            prediction = None

    if prediction is None:
        try:
            prediction = _call_grok(team_state, trace)
        except Exception as exc:
            if trace is not None:
                trace["source"] = "grok_error"
                trace["error"] = f"{type(exc).__name__}: {exc}"
                _write_trace(trace)
            raise

    if trace is not None:
        trace["final_prediction"] = prediction
        _write_trace(trace)
    return prediction
