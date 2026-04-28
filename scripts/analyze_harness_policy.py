#!/usr/bin/env python3
"""Print the active declarative harness policy.

Read-only. Does not call Grok and does not score features statistically. This
shows the agent-authored field groups currently used to curate the Grok payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper import harness_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print single JSON object to stdout",
    )
    args = parser.parse_args()

    # Clear module cache so env vars apply (e.g. MLB_HARNESS_POLICY_PATH in same run)
    if hasattr(harness_policy, "_POLICY_CACHE"):
        harness_policy._POLICY_CACHE = None  # type: ignore[attr-defined]

    pol = harness_policy.get_harness_policy()
    if args.json:
        print(json.dumps(pol, indent=2, sort_keys=True))
        return

    print("source:", pol.get("source"))
    if pol.get("notes"):
        print("notes:", pol.get("notes"))
    print("selected_groups:", ", ".join(pol.get("selected_groups", [])) or "(none)")
    print()
    print("--- available team groups ---")
    for group in pol.get("available_team_groups", []):
        print(" ", group)
    print()
    print("--- available player groups ---")
    for group in pol.get("available_player_groups", []):
        print(" ", f"player__{group}")
    print()
    print("--- team keys (", len(pol.get("team_keys", [])), ") ---", sep="")
    for k in pol.get("team_keys", []):
        print(" ", k)
    print()
    pl = harness_policy.selected_player_keys()
    print("--- player keys (", len(pl), ") ---", sep="")
    for k in pl:
        print(" ", k)


if __name__ == "__main__":
    main()
