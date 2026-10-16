#!/usr/bin/env python3
"""
Voice check: does the emotion label we send to CosyVoice change how the line sounds?

Live TTS only gets the spoken MSP label (Anger, Sadness, …) as a fixed instruct2
sentence. This harness synthesizes the same text under every label through the live
TTS path, scores each clip with speech-emotion models trained on MSP-Podcast (the
corpus our PAD anchors come from), and compares every label against the same line
spoken as Neutral.

  synth   controlled set: fixture lines × roles × labels × repeats
          (--session LOG adds every logged panel line with its live label + a Neutral twin)
  score   dimensional SER: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
          (arousal / dominance / valence; used to label emotional TTS in EmoSphere-TTS)
          categorical SER: 3loi/SER-Odyssey-Baseline-WavLM-Categorical
          (MSP-Podcast 8 classes; Odyssey 2024 SER challenge baseline)
  msp     add real MSP-Podcast clips (consensus label, Test1) as the reference shift
  report  per label vs its Neutral twin: mean Δ valence/arousal/dominance with 95 % CI,
          sign agreement with the label's anchor centroid, categorical hit rate and lift

Usage (compute node with the CosyVoice sidecar up; HF_HOME on /data):
    python eval/voice_emotion.py synth --out eval/results/voice_emotion/run1
    python eval/voice_emotion.py msp --out eval/results/voice_emotion/run1
    python eval/voice_emotion.py score --out eval/results/voice_emotion/run1 --model dim
    python eval/voice_emotion.py score --out eval/results/voice_emotion/run1 --model cat  # transformers 4.x env
    python eval/voice_emotion.py report --out eval/results/voice_emotion/run1
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

LABELS = ["Neutral", "Happiness", "Anger", "Sadness", "Surprise", "Fear", "Disgust", "Contempt"]
ROLES = ["host", "commentator", "guest"]
DIM_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
CAT_MODEL = "3loi/SER-Odyssey-Baseline-WavLM-Categorical"
# Odyssey class names → our MSP labels.
CAT_TO_LABEL = {
    "angry": "Anger", "anger": "Anger", "sad": "Sadness", "sadness": "Sadness",
    "happy": "Happiness", "happiness": "Happiness", "surprise": "Surprise",
    "fear": "Fear", "disgust": "Disgust", "contempt": "Contempt", "neutral": "Neutral",
}


# --- synth -------------------------------------------------------------------


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def _session_items(log: Path) -> list[dict[str, Any]]:
    """Every logged panel line with the label it was spoken with, plus a Neutral twin."""
    items: list[dict[str, Any]] = []
    for raw in log.read_text(encoding="utf-8").splitlines():
        d = json.loads(raw)
        if d.get("event") != "panel_model_done" or not d.get("reply"):
            continue
        label = (d.get("emotion") or "Neutral").strip()
        label = label[:1].upper() + label[1:]
        group = f"{log.stem}:{d.get('step')}"
        for lab in {label, "Neutral"}:
            items.append({"source": "session", "group": group, "role": d["role"],
                          "label": lab, "live_label": label, "text": d["reply"]})
    return items


# Instruct wordings to compare. "live" is agent.emotion.tts.emotion_instruct (what
# the show sends today); the others replace only the instruct sentence. Since
# 2026-10-16 live English = strong_en (the v1 report's "live" rows are the old
# mild wording).
_STRONG_EN = {
    "Neutral": "Speak in a calm, natural conversational tone.",
    "Happiness": "Speak very happily and cheerfully, with a bright, smiling, upbeat voice.",
    "Anger": "Speak very angrily, with a loud, sharp, tense and forceful voice.",
    "Sadness": "Speak very sadly, with a low, slow, heavy, tearful voice.",
    "Surprise": "Speak with great surprise and astonishment, in a high, excited voice.",
    "Fear": "Speak fearfully and nervously, with a trembling, anxious voice.",
    "Disgust": "Speak with strong disgust and revulsion in your voice.",
    "Contempt": "Speak with cold contempt, sneering and scornful.",
}
_STRONG_ZH = {
    "Neutral": "请用平静自然的语气说这句话。",
    "Happiness": "请用非常开心的语气说这句话。",
    "Anger": "请用非常生气的语气说这句话。",
    "Sadness": "请用非常伤心的语气说这句话。",
    "Surprise": "请用非常惊讶的语气说这句话。",
    "Fear": "请用非常害怕的语气说这句话。",
    "Disgust": "请用非常厌恶的语气说这句话。",
    "Contempt": "请用非常轻蔑的语气说这句话。",
}
VARIANTS = ("live", "zh", "strong_en", "strong_zh")


def variant_instruct(variant: str, label: str) -> str:
    from agent.emotion.tts import emotion_instruct

    if variant == "live":
        return emotion_instruct(emotion=label, locale="en")
    if variant == "zh":
        return emotion_instruct(emotion=label, locale="zh")
    body = (_STRONG_EN if variant == "strong_en" else _STRONG_ZH)[label]
    return f"You are a helpful assistant. {body}<|endofprompt|>"


def _controlled_items(lines: list[str], repeats: int, variant: str = "live") -> list[dict[str, Any]]:
    prefix = "" if variant == "live" else f"{variant}:"  # live keeps the v1 ids
    return [
        {"source": "controlled", "group": f"{prefix}l{i}:{role}:r{rep}", "role": role,
         "label": label, "text": text, "variant": variant}
        for i, text in enumerate(lines)
        for role in ROLES
        for rep in range(repeats)
        for label in LABELS
    ]


async def synth(
    out: Path, *, repeats: int, sessions: list[Path], concurrency: int, variants: list[str]
) -> None:
    import yaml
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
    from agent.adapters.cosyvoice_client import synthesize_pcm_via_sidecar
    from agent.adapters.cosyvoice_voices import cosyvoice_mode
    from agent.adapters.tts_synthesize import synthesize_pcm_for_config
    from agent.config import load_config, load_persona_tts

    lines = yaml.safe_load((_REPO / "eval/fixtures/voice_emotion_lines.yaml").read_text())["lines"]
    items = [it for v in variants for it in _controlled_items(lines, repeats, v)]
    for log in sessions:
        items += _session_items(log)
    app_cfg = load_config()
    cfgs = {r: load_persona_tts(r, app_cfg, locale="en") for r in {i["role"] for i in items}}
    (out / "wav").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.is_file():
        done = {json.loads(x)["id"] for x in manifest.read_text().splitlines()}
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(item: dict[str, Any]) -> None:
        item_id = f"{item['group']}:{item['label']}".replace("/", "_")
        if item_id in done:
            return
        async with sem:
            if item.get("variant", "live") == "live":
                pcm, rate, _ = await synthesize_pcm_for_config(
                    cfgs[item["role"]], item["text"], emotion=item["label"], locale="en",
                    role=item["role"], step="voice_emotion",
                )
            else:  # same voice + prompt wav as live; only the instruct differs
                pcm, rate, _ = await synthesize_pcm_via_sidecar(
                    text=item["text"], instruct=variant_instruct(item["variant"], item["label"]),
                    prompt_wav=cfgs[item["role"]].model_path, mode=cosyvoice_mode(),
                )
        path = out / "wav" / f"{hashlib.md5(item_id.encode()).hexdigest()[:12]}.wav"
        _write_wav(path, pcm, rate)
        async with lock:
            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": item_id, "wav": str(path.relative_to(out)),
                                    "seconds": round(len(pcm) / 2 / rate, 2), **item}) + "\n")
            done.add(item_id)
            if len(done) % 24 == 0:
                print(f"  {len(done)}/{len(items)} clips", flush=True)

    print(f"synthesizing {len(items)} clips → {out}", flush=True)
    await asyncio.gather(*(one(i) for i in items))
    print(f"done: {len(done)} clips", flush=True)


# --- score -------------------------------------------------------------------


def _load_audio(path: Path, rate: int) -> Any:
    import numpy as np
    import torch
    import torchaudio.functional as AF

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    t = torch.from_numpy(x)
    return AF.resample(t, sr, rate) if sr != rate else t


def _dim_model(device: str) -> Any:
    import torch
    from torch import nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config: Any) -> None:
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x: Any) -> Any:
            return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values: Any) -> Any:
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.classifier(hidden)

    model = EmotionModel.from_pretrained(DIM_MODEL).to(device).eval()
    return model


def _cat_model(device: str) -> Any:
    from transformers import AutoModelForAudioClassification

    return AutoModelForAudioClassification.from_pretrained(CAT_MODEL, trust_remote_code=True).to(device).eval()


def score(out: Path, *, device: str, model: str) -> None:
    """One SER model per pass (``scores_dim.jsonl`` / ``scores_cat.jsonl``), so the
    categorical model can run in an env whose transformers loads its remote code."""
    import torch

    rows = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    scored_path = out / f"scores_{model}.jsonl"
    have = set()
    if scored_path.is_file():
        have = {json.loads(x)["id"] for x in scored_path.read_text().splitlines()}
    todo = [r for r in rows if r["id"] not in have]
    print(f"scoring {len(todo)} clips with {model} on {device}", flush=True)
    net = _dim_model(device) if model == "dim" else _cat_model(device)
    id2label = {int(k): v for k, v in net.config.id2label.items()} if model == "cat" else {}
    with scored_path.open("a", encoding="utf-8") as f, torch.no_grad():
        for n, r in enumerate(todo, 1):
            wav = _load_audio(out / r["wav"], 16000)
            if model == "dim":
                x = (wav - wav.mean()) / (wav.std() + 1e-7)  # Wav2Vec2Processor normalisation
                a, d, v = net(x.unsqueeze(0).to(device))[0].tolist()
                rec: dict[str, Any] = {"id": r["id"], "valence": v, "arousal": a, "dominance": d}
            else:
                y = ((wav - net.config.mean) / (net.config.std + 1e-6)).unsqueeze(0).to(device)
                logits = net(y, torch.ones_like(y))
                logits = getattr(logits, "logits", logits)
                probs = torch.softmax(logits, dim=-1)[0].tolist()
                rec = {"id": r["id"], "cat": {
                    CAT_TO_LABEL.get(id2label[i].lower(), id2label[i]): p for i, p in enumerate(probs)
                }}
            f.write(json.dumps(rec) + "\n")
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print("done", flush=True)


def msp_reference(out: Path, *, per_label: int, csv: Path, audio_dir: Path, seed: int = 0) -> None:
    """Add real MSP-Podcast clips (consensus label, Test1) as a ceiling for the TTS shifts."""
    import csv as csvmod

    rng = random.Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    names = {"A": "Anger", "S": "Sadness", "H": "Happiness", "U": "Surprise", "F": "Fear",
             "D": "Disgust", "C": "Contempt", "N": "Neutral"}
    with csv.open(newline="") as f:
        reader = csvmod.DictReader(f)
        split_col = next((c for c in ("Split_Set", "Split", "split") if c in (reader.fieldnames or [])), None)
        print(f"  columns={reader.fieldnames} split={split_col}", flush=True)
        for row in reader:
            lab = names.get((row.get("EmoClass") or "").strip())
            split = (row.get(split_col) or "").strip() if split_col else "Test1"
            if lab and split == "Test1" and (audio_dir / row["FileName"]).is_file():
                by_label[lab].append(row["FileName"])
    manifest = out / "manifest.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    have = set()
    if manifest.is_file():
        have = {json.loads(x)["id"] for x in manifest.read_text().splitlines()}
    with manifest.open("a", encoding="utf-8") as f:
        for lab in LABELS:
            files = sorted(by_label[lab])
            for name in rng.sample(files, min(per_label, len(files))):
                item_id = f"msp:{name}"
                if item_id not in have:
                    f.write(json.dumps({"id": item_id, "wav": str(audio_dir / name), "source": "msp",
                                        "group": item_id, "label": lab, "role": ""}) + "\n")
            print(f"  {lab}: {min(per_label, len(files))} of {len(files)} Test1 clips", flush=True)


# --- report ------------------------------------------------------------------


def _ci(xs: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    boots = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(iters))
    return m, boots[int(0.025 * iters)], boots[int(0.975 * iters)]


def _anchor_direction() -> dict[str, tuple[float, float, float]]:
    """Label centroid − Neutral centroid in our anchor PAD space (P, A, D)."""
    import numpy as np

    from agent.emotion.pad_pipeline import get_pad_knn

    knn = get_pad_knn()
    if knn is None:
        return {}
    labels = np.array(knn.labels)
    neutral = knn.pad[labels == "Neutral"].mean(axis=0)
    return {lab: tuple(knn.pad[labels == lab].mean(axis=0) - neutral)
            for lab in LABELS if (labels == lab).any()}


def report(out: Path, *, source: str, variant: str = "live") -> dict[str, Any]:
    rows = {json.loads(x)["id"]: json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()}
    scores: dict[str, dict[str, Any]] = defaultdict(dict)
    for model in ("dim", "cat"):
        path = out / f"scores_{model}.jsonl"
        if path.is_file():
            for x in path.read_text().splitlines():
                rec = json.loads(x)
                scores[rec["id"]].update(rec)
    if source == "msp":
        return _report_unpaired(rows, scores, out)
    by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for i, r in rows.items():
        if i in scores and r["source"] == source and r.get("variant", "live") == variant:
            by_group[r["group"]][r["label"]] = {**r, **scores[i]}
    direction = _anchor_direction()
    has_cat = any("cat" in s for s in scores.values())
    axes = [("valence", 0), ("arousal", 1), ("dominance", 2)]
    summary: dict[str, Any] = {}
    print(f"\nsource={source} variant={variant}  groups={len(by_group)}  (Δ = label − same line as Neutral; SER scale 0–1)")
    head = f"{'label':<10}{'n':>4}" + "".join(f"{'Δ' + a[:3]:>22}" for a, _ in axes)
    head += f"{'sign ok':>9}" + (f"{'hit':>7}{'lift':>8}" if has_cat else "")
    print(head)
    for lab in LABELS:
        pairs = [(g[lab], g["Neutral"]) for g in by_group.values() if lab in g and "Neutral" in g]
        if not pairs:
            continue
        s: dict[str, Any] = {"n": len(pairs)}
        line = f"{lab:<10}{len(pairs):>4}"
        agree = []
        for axis, k in axes:
            m, lo, hi = _ci([x[axis] - n[axis] for x, n in pairs])
            s[axis] = {"mean": m, "ci": [lo, hi]}
            line += f"{m:>+10.3f} [{lo:+.2f},{hi:+.2f}]"
            want = direction.get(lab, (0, 0, 0))[k]
            if lab != "Neutral" and abs(want) > 0.05:
                agree.append((m > 0) == (want > 0) and (lo > 0 or hi < 0))
        s["sign_ok"] = f"{sum(agree)}/{len(agree)}" if agree else "-"
        line += f"{s['sign_ok']:>9}"
        if has_cat:
            hit = sum(max(x["cat"], key=x["cat"].get) == lab for x, _ in pairs) / len(pairs)
            lift = sum(x["cat"].get(lab, 0) - n["cat"].get(lab, 0) for x, n in pairs) / len(pairs)
            s["hit"], s["lift"] = hit, lift
            line += f"{hit:>7.0%}{lift:>+8.3f}"
        summary[lab] = s
        print(line)
    if direction:
        print("\nexpected direction (anchor centroid − Neutral, P/A/D):")
        for lab, d in direction.items():
            print(f"  {lab:<10} " + " ".join(f"{x:+.2f}" for x in d))
    name = source if variant == "live" else f"{source}_{variant}"
    (out / f"report_{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def _report_unpaired(rows: dict[str, Any], scores: dict[str, Any], out: Path) -> dict[str, Any]:
    """Real speech has no Neutral twin: label mean − Neutral mean, bootstrap CI."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, r in rows.items():
        if r["source"] == "msp" and "valence" in scores.get(i, {}):
            by_label[r["label"]].append(scores[i])
    neutral = by_label.get("Neutral", [])
    rng = random.Random(0)
    summary: dict[str, Any] = {}
    print(f"\nsource=msp  (Δ = label mean − Neutral mean over real MSP-Podcast clips)")
    print(f"{'label':<10}{'n':>4}" + "".join(f"{'Δ' + a[:3]:>22}" for a in ("valence", "arousal", "dominance")))
    for lab in LABELS:
        xs = by_label.get(lab, [])
        if not xs or not neutral:
            continue
        s: dict[str, Any] = {"n": len(xs)}
        line = f"{lab:<10}{len(xs):>4}"
        for axis in ("valence", "arousal", "dominance"):
            m = sum(x[axis] for x in xs) / len(xs) - sum(x[axis] for x in neutral) / len(neutral)
            boots = sorted(
                sum(x[axis] for x in rng.choices(xs, k=len(xs))) / len(xs)
                - sum(x[axis] for x in rng.choices(neutral, k=len(neutral))) / len(neutral)
                for _ in range(2000)
            )
            s[axis] = {"mean": m, "ci": [boots[50], boots[1949]]}
            line += f"{m:>+10.3f} [{boots[50]:+.2f},{boots[1949]:+.2f}]"
        summary[lab] = s
        print(line)
    (out / "report_msp.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synth")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--session", type=Path, action="append", default=[])
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--variants", default="live", help=f"comma list of {VARIANTS}")
    p = sub.add_parser("score")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    p.add_argument("--model", default="dim", choices=("dim", "cat"))
    p = sub.add_parser("msp")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-label", type=int, default=60)
    p.add_argument("--csv", type=Path, default=Path(
        "/data/user_data/hsuanhal/MSP-PODCAST-Publish-2.0/Labels/labels_consensus.csv"))
    p.add_argument("--audio-dir", type=Path, default=Path(
        "/data/user_data/hsuanhal/MSP-PODCAST-Publish-2.0/Audios"))
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", default="controlled", choices=("controlled", "session", "msp"))
    p.add_argument("--variant", default="live", choices=VARIANTS)
    a = ap.parse_args()
    if a.cmd == "synth":
        asyncio.run(synth(a.out, repeats=a.repeats, sessions=a.session, concurrency=a.concurrency,
                          variants=[v.strip() for v in a.variants.split(",") if v.strip()]))
    elif a.cmd == "score":
        score(a.out, device=a.device, model=a.model)
    elif a.cmd == "msp":
        msp_reference(a.out, per_label=a.per_label, csv=a.csv, audio_dir=a.audio_dir)
    else:
        report(a.out, source=a.source, variant=a.variant)


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
