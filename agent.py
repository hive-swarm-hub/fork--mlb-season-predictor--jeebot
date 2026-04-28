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

from helper.features import clamp
from helper.harness_policy import get_harness_policy, selected_player_keys, selected_team_keys
from helper.league_context import peer_summary


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache" / "grok_predictions"
DEFAULT_MODEL = "grok-4-1-fast-reasoning"


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


def _sample_cache_key(team_state: dict, model: str, sample_idx: int) -> str:
    payload = {
        "harness": _harness_fingerprint(),
        "model": model,
        "sample_idx": sample_idx,
        "prompt_payload": _prompt_payload(team_state),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _single_grok_sample(team_state: dict, sample_idx: int = 0) -> dict:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is required; this task has no local fallback")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required to call the Grok-compatible API") from exc
    model = os.getenv("XAI_MODEL", DEFAULT_MODEL)
    if sample_idx == 0:
        key = _cache_key(team_state, model)
    else:
        key = _sample_cache_key(team_state, model, sample_idx)
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        return _normalize_prediction(json.loads(cache_path.read_text()))
    client = OpenAI(api_key=api_key, base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"))
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
    content = response.choices[0].message.content or "{}"
    raw = _extract_json(content)
    prediction = _normalize_prediction(raw)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prediction, sort_keys=True) + "\n")
    return prediction


def _aggregate_predictions(samples: list[dict]) -> dict:
    """Mean of projected_wins, median of probabilities, median of interval bounds."""
    import statistics
    wins = statistics.fmean([s["projected_wins"] for s in samples])
    lows = statistics.median([s["win_interval_80"][0] for s in samples])
    highs = statistics.median([s["win_interval_80"][1] for s in samples])
    return _normalize_prediction({
        "projected_wins": wins,
        "playoff_prob": statistics.median([s["playoff_prob"] for s in samples]),
        "division_winner_prob": statistics.median([s["division_winner_prob"] for s in samples]),
        "league_champion_prob": statistics.median([s["league_champion_prob"] for s in samples]),
        "world_series_champion_prob": statistics.median([s["world_series_champion_prob"] for s in samples]),
        "win_interval_80": [lows, highs],
    })


def _call_grok(team_state: dict, trace: dict | None = None) -> dict:
    n_samples = max(1, int(os.getenv("MLB_N_SAMPLES", "3")))
    model = os.getenv("XAI_MODEL", DEFAULT_MODEL)
    if trace is not None:
        trace["model"] = model
        trace["n_samples"] = n_samples
        trace["prompt_payload"] = _prompt_payload(team_state)
    samples = [_single_grok_sample(team_state, i) for i in range(n_samples)]
    prediction = _aggregate_predictions(samples) if len(samples) > 1 else samples[0]
    if trace is not None:
        trace["source"] = "grok_self_consistency" if n_samples > 1 else "grok_api"
        trace["model_prediction"] = prediction
        trace["sample_predictions"] = samples
    return prediction

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
