"""Feature helpers for the starter MLB season predictor.

Agents are expected to replace or extend these helpers with richer baseball
features. Keep eval-time code self-contained: no network calls here.
"""

from __future__ import annotations

import csv
from pathlib import Path


NUMERIC_FIELDS = (
    "age",
    "proj_pa",
    "proj_ip",
    "prev_pa",
    "prev_ip",
    "steamer_war",
    "zips_war",
    "dc_war",
    "projection_blend_war",
    "woba",
    "xwoba",
    "barrel_pct",
    "hard_hit_pct",
    "exit_velocity",
    "chase_pct",
    "stuff_plus",
    "location_plus",
    "projected_leverage_index",
    "framing_runs",
    "drs_runs",
    "uzr_runs",
    "oaa_runs",
    "def_runs",
    "durability_score",
    "injury_risk",
    "dc_team_war",
    "steamer_team_war",
    "zips_team_war",
    "projection_blend_war",
    "pos_war",
    "sp_war",
    "rp_war",
    "sp1_war",
    "sp2_war",
    "sp3_war",
    "sp4_war",
    "sp5_war",
    "sp6_war",
    "sp7_war",
    "bp_high_lev_war",
    "rotation_depth_war",
    "catcher_framing_runs",
    "drs_runs",
    "uzr_runs",
    "oaa_runs",
    "def_blend_runs",
    "schedule_strength",
    "division_strength",
    "intradivision_schedule_strength",
    "park_factor",
    "park_factor_3yr",
    "park_factor_1yr",
    "injury_war_lost",
    "durability_risk",
    "age_risk",
    "pythag_win_pct",
    "baseruns_win_pct",
    "third_order_win_pct",
    "prev_win_pct",
    "overall_rank",
    "league_rank",
    "division_rank",
    "actual_wins",
    "made_playoffs",
    "won_division",
    "league_champion",
    "world_series_champion",
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def coerce_row(row: dict) -> dict:
    out = dict(row)
    for field in NUMERIC_FIELDS:
        if field in out and out[field] != "":
            out[field] = float(out[field])
    if "season" in out:
        out["season"] = int(float(out["season"]))
    return out


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return [coerce_row(row) for row in csv.DictReader(f)]


def roster_key(row: dict) -> tuple[int, str, str]:
    return (int(row["season"]), str(row["checkpoint"]), str(row["team_id"]))


def load_rosters(path: str | Path) -> dict[tuple[int, str, str], list[dict]]:
    rosters: dict[tuple[int, str, str], list[dict]] = {}
    if not Path(path).exists():
        return rosters
    for row in load_rows(path):
        rosters.setdefault(roster_key(row), []).append(row)
    return rosters
