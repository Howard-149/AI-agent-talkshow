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
#   python cli/text_session_control.py human --job SLURM_JOB_ID --text "What about rural schools?"

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "start",
            "stop",
            "restart",
            "status",
            "human",
        ],
    )

    parser.add_argument(
        "--job",
        required=True,
    )

    parser.add_argument(
        "--text",
        default="",
        help="human: the line to say (interrupts the current speaker)",
    )

    args = parser.parse_args()
    if args.command == "human" and not args.text.strip():
        parser.error("human requires --text")

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
                "text": args.text,
            }
        )
    )

    os.replace(
        tmp,
        target,
    )


if __name__ == "__main__":
    main()