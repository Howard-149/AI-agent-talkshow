"""Pad the square role portraits to 1920x1080 for AVTR-1, which resizes any portrait to
16:9 (a square one came out stretched 1.78x wide). The portrait sits in the centre at
full height; the sides repeat its edge pixels and are cropped away by the sidecar.

  pixi run python deploy/avtr1/prepare_portraits.py <storage>/main/avatars_artifacts/reference_frames
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
# AVTR-1 avatar id → portrait (avatar/avtr1_sidecar.py maps roles and file names to these ids)
PORTRAITS = {"claire": "lessac.png", "ryan": "ryan.png", "amy": "amy.png"}


def main() -> None:
    dst = Path(sys.argv[1])
    for avatar_id, name in PORTRAITS.items():
        img = cv2.imread(str(ROOT / "avatar/assets/portraits" / name), cv2.IMREAD_COLOR)
        shrink = img.shape[0] > 1080
        img = cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
        side = (1920 - 1080) // 2
        cv2.imwrite(str(dst / f"{avatar_id}.png"), cv2.copyMakeBorder(img, 0, 0, side, side, cv2.BORDER_REPLICATE))
        print(avatar_id, "→", dst / f"{avatar_id}.png")


if __name__ == "__main__":
    main()
