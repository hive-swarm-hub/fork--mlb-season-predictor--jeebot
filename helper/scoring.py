"""Shared scoring helpers for dev and frozen evaluation."""

from __future__ import annotations

import math
from typing import Any

from helper.features import clamp

LABEL_COLUMNS = {
    "actual_wins",
    "overall_rank",
    "league_rank",
    "division_rank",
    "made_playoffs",
    "won_division",
    "league_champion",
    "world_series_champion",
}
LABEL_KEY_COLUMNS = ("season", "checkpoint", "team_id")


def label_key(row: dict) -> tuple[int, str, str]:
    return (int(row["season"]), str(row["checkpoint"]), str(row["team_id"]))


def split_features_and_labels(rows: list[dict]) -> tuple[list[dict], dict[tuple[int, str, str], dict]]:
    features: list[dict] = []
    labels: dict[tuple[int, str, str], dict] = {}
    for row in rows:
        features.append({k: v for k, v in row.items() if k not in LABEL_COLUMNS})
        labels[label_key(row)] = {
            k: row[k] for k in (*LABEL_KEY_COLUMNS, *LABEL_COLUMNS) if k in row
        }
    return features, labels


def binary_loss(prob: float, label: int) -> float:
    p = clamp(float(prob), 1e-6, 1.0 - 1e-6)
    y = int(label)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def assign_predicted_ranks(scored_rows: list[dict]) -> None:
    groups: dict[tuple[int, str], list[dict]] = {}
    for item in scored_rows:
        row = item["row"]
        groups.setdefault((int(row["season"]), str(row["checkpoint"])), []).append(item)

    for group_items in groups.values():
        overall = sorted(
            group_items,
            key=lambda item: (-float(item["projected_wins"]), str(item["row"]["team_id"])),
        )
        for rank, item in enumerate(overall, start=1):
            item["pred_overall_rank"] = rank

        leagues = sorted({item["row"]["league"] for item in group_items})
        for league in leagues:
            league_items = [item for item in overall if item["row"]["league"] == league]
            for rank, item in enumerate(league_items, start=1):
                item["pred_league_rank"] = rank

        divisions = sorted({item["row"]["division"] for item in group_items})
        for division in divisions:
            division_items = [
                item for item in overall if item["row"]["division"] == division
            ]
            for rank, item in enumerate(division_items, start=1):
                item["pred_division_rank"] = rank


def score_rows(scored_rows: list[dict]) -> dict[str, Any]:
    assign_predicted_ranks(scored_rows)

    league_rank_abs_error = 0.0
    overall_rank_abs_error = 0.0
    division_rank_abs_error = 0.0
    win_abs_error = 0.0
    league_champion_loss = 0.0
    world_series_loss = 0.0
    playoff_loss = 0.0
    exact_league_rank = 0

    for item in scored_rows:
        label = item["label"]
        pred = item["pred"]
        projected_wins = item["projected_wins"]
        league_rank_abs_error += abs(
            int(item["pred_league_rank"]) - int(label["league_rank"])
        )
        overall_rank_abs_error += abs(
            int(item["pred_overall_rank"]) - int(label["overall_rank"])
        )
        division_rank_abs_error += abs(
            int(item["pred_division_rank"]) - int(label["division_rank"])
        )
        exact_league_rank += int(int(item["pred_league_rank"]) == int(label["league_rank"]))
        win_abs_error += abs(projected_wins - float(label["actual_wins"]))
        league_champion_loss += binary_loss(
            pred.get("league_champion_prob", 1.0 / 15.0), int(label["league_champion"])
        )
        world_series_loss += binary_loss(
            pred.get("world_series_champion_prob", 1.0 / 30.0),
            int(label["world_series_champion"]),
        )
        playoff_loss += binary_loss(pred.get("playoff_prob", 0.5), int(label["made_playoffs"]))

    total = len(scored_rows)
    if total == 0:
        raise ValueError("cannot score zero rows")

    rank_mae = league_rank_abs_error / total
    overall_rank_mae = overall_rank_abs_error / total
    division_rank_mae = division_rank_abs_error / total
    league_champ_log_loss = league_champion_loss / total
    ws_champ_log_loss = world_series_loss / total
    playoff_log_loss = playoff_loss / total
    win_mae = win_abs_error / total
    composite_loss = (
        0.50 * rank_mae
        + 0.15 * overall_rank_mae
        + 0.10 * division_rank_mae
        + 0.10 * (win_mae / 5.0)
        + 0.06 * league_champ_log_loss
        + 0.06 * ws_champ_log_loss
        + 0.03 * playoff_log_loss
    )
    return {
        "score": -composite_loss,
        "composite_loss": composite_loss,
        "rank_mae": rank_mae,
        "overall_rank_mae": overall_rank_mae,
        "division_rank_mae": division_rank_mae,
        "league_champ_log_loss": league_champ_log_loss,
        "ws_champ_log_loss": ws_champ_log_loss,
        "playoff_log_loss": playoff_log_loss,
        "win_mae": win_mae,
        "correct": exact_league_rank,
        "total": total,
    }


def print_score_report(metrics: dict[str, Any], *, prefix: str | None = None) -> None:
    print("---")
    if prefix:
        print(prefix)
    print(f"score:            {metrics['score']:.4f}")
    print(f"composite_loss:   {metrics['composite_loss']:.4f}")
    print(f"rank_mae:         {metrics['rank_mae']:.4f}")
    print(f"overall_rank_mae: {metrics['overall_rank_mae']:.4f}")
    print(f"division_rank_mae:{metrics['division_rank_mae']:.4f}")
    print(f"league_champ_log_loss: {metrics['league_champ_log_loss']:.4f}")
    print(f"ws_champ_log_loss:{metrics['ws_champ_log_loss']:.4f}")
    print(f"playoff_log_loss: {metrics['playoff_log_loss']:.4f}")
    print(f"win_mae:          {metrics['win_mae']:.1f}")
    print(f"correct:          {metrics['correct']}")
    print(f"total:            {metrics['total']}")
