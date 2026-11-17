"""Which other female MSP-Podcast speakers could voice a role? For each female speaker with
enough clips of every emotion, pick one clip per emotion (3-12 s) that sounds most like the
speaker's own Neutral (CosyVoice's campplus embedding) and report how far apart the picks
sound: a small gap means the cloned voice stays the same person across moods.

Run on a Babel compute node (MSP-Podcast mounted):

  python eval/msp_female_candidates.py [--top 8] [--copy-to logs/msp-candidates] > logs/msp-female-candidates.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((REPO / "config" / "msp_emotion_refs.yaml").read_text())
ROOT = Path(CFG["msp_root"])
CLASS = {"N": "Neutral", "H": "Happiness", "A": "Anger", "S": "Sadness", "U": "Surprise"}
NEEDED = ("Neutral", "Happiness", "Anger", "Sadness")  # Surprise falls back to Neutral if missing
MIN_S, MAX_S = 3.0, 12.0
PER_CLASS = 25  # clips embedded per emotion per speaker (cost bound)


def duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def make_embedder(model: str):
    """campplus speaker embedding, same features as eval/msp_ref_similarity.py."""
    import onnxruntime as ort
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])

    def embed(path: Path) -> np.ndarray:
        with wave.open(str(path)) as w:
            sr, ch = w.getframerate(), w.getnchannels()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        speech = torch.from_numpy(x.reshape(-1, ch).mean(axis=1)).unsqueeze(0)
        if sr != 16000:
            speech = torchaudio.functional.resample(speech, sr, 16000)
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        v = sess.run(None, {sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
        return v / np.linalg.norm(v)

    return embed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8, help="speakers to embed (most clips first)")
    ap.add_argument("--min-per-class", type=int, default=4)
    ap.add_argument("--exclude", default="", help="comma-separated speaker ids already auditioned")
    ap.add_argument("--copy-to", type=Path, default=None, help="copy each pick here to listen")
    ap.add_argument("--model", default="/data/user_data/hsuanhal/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx")
    args = ap.parse_args()

    embed = make_embedder(args.model)

    used = {str(spec["speaker"]) for spec in CFG["roles"].values()}
    used |= {x.strip() for x in args.exclude.split(",") if x.strip()}
    rows = list(csv.DictReader(open(ROOT / "Labels" / "labels_consensus.csv")))
    by_spk: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("Gender", "").lower() != "female" or r["EmoClass"] not in CLASS:
            continue
        by_spk[r["SpkrID"]][CLASS[r["EmoClass"]]].append(r)

    def coverage(spk: str) -> int:
        return min(len(by_spk[spk][e]) for e in NEEDED)

    eligible = [s for s in by_spk if s not in ("Unknown", "") and coverage(s) >= args.min_per_class]
    eligible.sort(key=coverage, reverse=True)
    current_guest = str(CFG["roles"]["guest"]["speaker"])
    pool = [s for s in eligible if s not in used][: args.top]
    if current_guest in by_spk:
        pool.append(current_guest)  # baseline: today's Amy, picked by the same rule
    print(f"female speakers eligible={len(eligible)} embedding={pool}", file=sys.stderr)

    report = []
    for spk in pool:
        clips: dict[str, list[tuple[dict, np.ndarray, float]]] = {}
        for emo, items in by_spk[spk].items():
            kept = []
            for r in items:
                path = ROOT / "Audios" / r["FileName"]
                if not path.is_file():
                    continue
                d = duration(path)
                if MIN_S <= d <= MAX_S:
                    kept.append((r, path, d))
                if len(kept) >= PER_CLASS:
                    break
            clips[emo] = [(r, embed(p), d) for r, p, d in kept]
        if any(not clips.get(e) for e in NEEDED):
            continue
        centroid = np.mean([v for _, v, _ in clips["Neutral"]], axis=0)
        centroid /= np.linalg.norm(centroid)
        neutral = max(clips["Neutral"], key=lambda c: float(c[1] @ centroid))
        anchor = neutral[1]
        picks = {"Neutral": {"file": neutral[0]["FileName"], "sim": 1.0, "sec": round(neutral[2], 1)}}
        for emo in ("Happiness", "Anger", "Sadness", "Surprise"):
            if not clips.get(emo):
                continue
            best = max(clips[emo], key=lambda c: float(c[1] @ anchor))
            picks[emo] = {
                "file": best[0]["FileName"],
                "sim": round(float(best[1] @ anchor), 3),
                "sec": round(best[2], 1),
                "act": round(float(best[0]["EmoAct"]), 2),
                "val": round(float(best[0]["EmoVal"]), 2),
            }
        sims = [p["sim"] for e, p in picks.items() if e != "Neutral"]
        report.append({
            "speaker": spk,
            "current_amy": spk == current_guest,
            "clips_per_emotion": {e: len(v) for e, v in by_spk[spk].items()},
            "min_sim": round(min(sims), 3),
            "mean_sim": round(float(np.mean(sims)), 3),
            "picks": picks,
        })
        print(f"speaker {spk}: min_sim={min(sims):.3f} mean={np.mean(sims):.3f}", file=sys.stderr)
        if args.copy_to is not None:
            out = args.copy_to / f"speaker{spk}"
            out.mkdir(parents=True, exist_ok=True)
            for emo, p in picks.items():
                shutil.copy(ROOT / "Audios" / p["file"], out / f"{emo}.wav")

    report.sort(key=lambda x: (-x["min_sim"], -x["mean_sim"]))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
