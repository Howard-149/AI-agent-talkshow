"""How alike do a role's MSP emotion references sound? Speaker similarity (CosyVoice's own
campplus embedding) of each reference to the role's Neutral reference, plus the best
same-speaker alternatives per emotion. Run on a Babel compute node:

  python eval/msp_ref_similarity.py > logs/msp-ref-similarity.json
"""
from __future__ import annotations

import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

REPO = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((REPO / "config" / "msp_emotion_refs.yaml").read_text())
ROOT = Path(CFG["msp_root"])
MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else
             "/data/user_data/hsuanhal/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx")
CLASS = {"N": "Neutral", "H": "Happiness", "A": "Anger", "S": "Sadness", "U": "Surprise"}
MIN_S, MAX_S = 3.0, 12.0

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
sess = ort.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])


def duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def embed(path: Path) -> np.ndarray:
    """Same features CosyVoice uses for its speaker embedding (16 kHz, 80-bin fbank, CMN)."""
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    speech = torch.from_numpy(x.reshape(-1, ch).mean(axis=1)).unsqueeze(0)
    if sr != 16000:
        speech = torchaudio.functional.resample(speech, sr, 16000)
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    out = sess.run(None, {sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0]
    v = out.flatten()
    return v / np.linalg.norm(v)


rows = list(csv.DictReader(open(ROOT / "Labels" / "labels_consensus.csv")))
report = {}
for role, spec in CFG["roles"].items():
    spk = str(spec["speaker"])
    refs = spec["refs"]
    anchor = embed(ROOT / "Audios" / refs["Neutral"])
    current = {emo: round(float(embed(ROOT / "Audios" / name) @ anchor), 3) for emo, name in refs.items()}
    pool = [r for r in rows if r["SpkrID"] == spk and r["EmoClass"] in CLASS]
    alternatives = {}
    for emo in refs:
        if emo == "Neutral":
            continue
        scored = []
        for r in pool:
            if CLASS[r["EmoClass"]] != emo:
                continue
            path = ROOT / "Audios" / r["FileName"]
            if not path.is_file():
                continue
            d = duration(path)
            if not MIN_S <= d <= MAX_S:
                continue
            scored.append({
                "file": r["FileName"],
                "sim": round(float(embed(path) @ anchor), 3),
                "sec": round(d, 1),
                "act": round(float(r["EmoAct"]), 2),
                "val": round(float(r["EmoVal"]), 2),
            })
        scored.sort(key=lambda x: -x["sim"])
        alternatives[emo] = scored[:5]
    report[role] = {"speaker": spk, "current_sim_to_neutral": current, "alternatives": alternatives}
    print(f"{role}: {current}", file=sys.stderr)

print(json.dumps(report, indent=1))
