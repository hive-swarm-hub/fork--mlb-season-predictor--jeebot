#!/usr/bin/env python3
"""Run local development evaluation on labeled train/val rows."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helper.features import clamp, load_rosters, load_rows, roster_key  # noqa: E402
from helper.scoring import label_key, print_score_report, score_rows  # noqa: E402


def load_agent():
    spec = importlib.util.spec_from_file_location("agent", ROOT / "agent.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise RuntimeError("agent.py must define predict(team_state)")
    return module


def load_split(split: str) -> tuple[list[dict], dict[tuple[int, str, str], list[dict]]]:
    team_path = ROOT / "data" / split / "team_states.csv"
    roster_path = ROOT / "data" / split / "player_states.csv"
    if not team_path.exists() or not roster_path.exists():
        raise SystemExit(f"missing {split} data; run bash prepare.sh first")
    return load_rows(team_path), load_rosters(roster_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="labeled public split to score",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional row limit for quick smoke runs",
    )
    args = parser.parse_args()

    agent = load_agent()
    rows, rosters = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]

    scored_rows = []
    for row in rows:
        team_state = dict(row)
        team_state["roster"] = rosters.get(roster_key(row), [])
        pred = agent.predict(team_state)
        scored_rows.append(
            {
                "row": row,
                "label": row,
                "pred": pred,
                "projected_wins": clamp(float(pred["projected_wins"]), 40.0, 122.0),
            }
        )

    metrics = score_rows(scored_rows)
    print_score_report(metrics, prefix=f"split:            {args.split}")
    print(f"rows_scored:      {len(rows)}")
    print(f"first_key:        {label_key(rows[0]) if rows else '(none)'}")


if __name__ == "__main__":
    main()
