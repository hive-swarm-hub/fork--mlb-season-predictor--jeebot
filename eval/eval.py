from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "eval" / "test_data" / "frozen_test.csv"
PUBLIC_LABEL_PATH = ROOT / "eval" / "frozen_labels.csv"
PRIVATE_LABEL_PATH = ROOT / "eval" / ".frozen_labels.csv"
ROSTER_PATH = ROOT / "eval" / "test_data" / "frozen_test_players.csv"
EVAL_SEASONS = {16}
sys.path.insert(0, str(ROOT))

from helper.features import clamp, load_rosters, load_rows, roster_key  # noqa: E402
from helper.scoring import (  # noqa: E402
    LABEL_COLUMNS,
    label_key,
    print_score_report,
    score_rows,
    split_features_and_labels,
)


def load_agent():
    spec = importlib.util.spec_from_file_location("agent", ROOT / "agent.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise RuntimeError("agent.py must define predict(team_state)")
    return module


def load_frozen_features_and_labels() -> tuple[list[dict], dict[tuple[int, str, str], dict]]:
    """Load frozen features and labels.

    The current public task ships labels for transparency. Private-label and
    legacy labeled-frozen files remain supported for local compatibility.
    """
    rows = load_rows(TEST_PATH)
    label_path = PUBLIC_LABEL_PATH if PUBLIC_LABEL_PATH.exists() else PRIVATE_LABEL_PATH
    if label_path.exists():
        labels = {label_key(row): row for row in load_rows(label_path)}
        features = [{k: v for k, v in row.items() if k not in LABEL_COLUMNS} for row in rows]
        return _filter_eval_seasons(features, labels)
    if rows and LABEL_COLUMNS.issubset(rows[0].keys()):
        features, labels = split_features_and_labels(rows)
        return _filter_eval_seasons(features, labels)
    raise SystemExit("missing frozen labels; expected eval/frozen_labels.csv")


def _filter_eval_seasons(
    features: list[dict], labels: dict[tuple[int, str, str], dict]
) -> tuple[list[dict], dict[tuple[int, str, str], dict]]:
    features = [row for row in features if int(row["season"]) in EVAL_SEASONS]
    keys = {label_key(row) for row in features}
    labels = {key: row for key, row in labels.items() if key in keys}
    return features, labels


def main() -> None:
    if not TEST_PATH.exists():
        raise SystemExit("missing frozen test data; run bash prepare.sh first")

    agent = load_agent()
    rows, labels = load_frozen_features_and_labels()
    rosters = load_rosters(ROSTER_PATH)
    scored_rows = []

    for row in rows:
        team_state = dict(row)
        team_state["roster"] = rosters.get(roster_key(row), [])
        pred = agent.predict(team_state)
        projected_wins = clamp(float(pred["projected_wins"]), 40.0, 122.0)
        label = labels.get(label_key(row))
        if label is None:
            raise RuntimeError(f"missing frozen label for {label_key(row)}")
        scored_rows.append(
            {
                "row": row,
                "label": label,
                "pred": pred,
                "projected_wins": projected_wins,
            }
        )

    metrics = score_rows(scored_rows)
    print_score_report(metrics, prefix=f"eval_seasons:     {','.join(map(str, sorted(EVAL_SEASONS)))}")


if __name__ == "__main__":
    main()
