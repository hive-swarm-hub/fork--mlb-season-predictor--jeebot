#!/usr/bin/env python3
"""Public placeholder for MLB Season Predictor data bundle generation.

The real bundle builder is intentionally not shipped with the agent-facing task
because it contains benchmark-sensitive details: real team identifiers, the
private anonymization map, and frozen labels. Keeping those out of the public
repo prevents agents from reconstructing TEAM_XX identities or reading answers.

Task maintainers should build the release artifact in a private workspace, then
publish only the public zip consumed by `prepare.sh`:

    train/team_states.csv
    train/player_states.csv
    val/team_states.csv
    val/player_states.csv
    frozen/frozen_test.csv           # feature columns only
    frozen/frozen_test_players.csv

The evaluator's private runtime must also provide:

    eval/frozen_labels.csv

Do not upload the private identity map as part of the task package.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "The sensitive MLB bundle builder is private. "
        "Publish a feature-only data zip and keep the real identity map private."
    )


if __name__ == "__main__":
    main()
