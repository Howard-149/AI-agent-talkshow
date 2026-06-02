#!/usr/bin/env python3
"""Dev helper: mint a LiveKit room token (run on laptop)."""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LiveKit access token")
    parser.add_argument("--room", default="talkshow-dev", help="Room name")
    parser.add_argument("--identity", default="human-host", help="Participant identity")
    parser.add_argument("--name", default=None, help="Display name (defaults to identity)")
    parser.add_argument("--ttl", type=int, default=3600, help="TTL seconds")
    args = parser.parse_args()

    display = args.name or args.identity

    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]

    from livekit import api

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(args.identity)
        .with_name(display)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=args.room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .with_ttl(timedelta(seconds=args.ttl))
        .to_jwt()
    )

    print(f"LIVEKIT_URL={url}")
    print(f"ROOM={args.room}")
    print(f"TOKEN={token}")


if __name__ == "__main__":
    main()
