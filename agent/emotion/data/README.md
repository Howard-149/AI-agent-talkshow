# MSP-PODCAST PAD anchors

Build on Babel (CSV lives under `/data/user_data/...`):

```bash
PYTHONPATH=. python eval/msp_podcast/inspect_and_build_anchors.py \
  --csv /data/user_data/hsuanhal/MSP-PODCAST-Publish-2.0/Labels/labels_detailed.csv \
  --out agent/emotion/data/msp_pad_anchors.npz
```

If `labels_detailed.csv` uses per-annotator rows (`WorkerID`, `EmoClass_Major`),
anchors will be ~1.4M points. For Sentipolis-like consensus (~265k), prefer:

```bash
... --csv .../Labels/labels_consensus.csv
```

Paper (Sentipolis A.2): normalize 1–7 → [-1,1], KNN k=3 Euclidean, keep all neighbor labels.

Live path (`TALKSHOW_EMOTION_SOURCE=pad`, default): load this NPZ via
`agent/emotion/pad_pipeline.py`, map PAD → MSP `primary_label` → `role_emotion`
→ CosyVoice `emotion_instruct` (open vocabulary; not the old closed-set tags).
