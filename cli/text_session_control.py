"""CLI tool for controlling text-based sessions at agent/session/text_session.py."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Command Examples:
#   python cli/text_session_control.py start --job SLURM_JOB_ID
#   python cli/text_session_control.py stop --job SLURM_JOB_ID
#   python cli/text_session_control.py restart --job SLURM_JOB_ID
#   python cli/text_session_control.py status --job SLURM_JOB_ID

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "start",
            "stop",
            "restart",
            "status",
        ],
    )

    parser.add_argument(
        "--job",
        required=True,
    )

    args = parser.parse_args()

    control_dir = Path(
        f"logs/text-control/{args.job}"
    )

    if args.command == "status":
        status_path = control_dir / "status.json"

        if not status_path.exists():
            print("text session manager not found")
            return

        print(status_path.read_text())
        return

    control_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = control_dir / "command.tmp"
    target = control_dir / "command.json"

    tmp.write_text(
        json.dumps(
            {
                "command": args.command,
            }
        )
    )

    os.replace(
        tmp,
        target,
    )


if __name__ == "__main__":
    main()