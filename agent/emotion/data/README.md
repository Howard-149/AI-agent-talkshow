# MSP-PODCAST PAD anchors

Build on a Babel **compute node** (`/data/user_data/...` is not mounted on login):

```bash
PYTHONPATH=. python eval/msp_podcast/inspect_and_build_anchors.py \
  --csv /data/user_data/hsuanhal/MSP-PODCAST-Publish-2.0/Labels/labels_consensus.csv \
  --out agent/emotion/data/msp_pad_anchors.npz
```

or `sbatch deploy/slurm-build-msp-anchors.sh`.

Use **`labels_consensus.csv`**: one row per utterance, attributes averaged across
annotators — 264,705 rows, exactly Sentipolis A.2. Do not use `labels_detailed.csv`
(one row per annotator, integer 1–7 ratings): it yields ~1.4M anchors on only 343
distinct PAD points, so k=3 "nearest" neighbors are arbitrary picks among thousands of
ties. The builder refuses per-annotator tables unless `--allow-per-annotator`.

Paper (Sentipolis A.2): normalize to [-1,1] (MSP attributes are 1–7, center 4 → 0),
KNN k=3 Euclidean, keep all neighbor labels; Other / No Agreement → Vague.

Live path (`TALKSHOW_EMOTION_SOURCE=pad`, default): load this NPZ via
`agent/emotion/pad_pipeline.py`, map PAD → MSP label → `role_emotion`
→ CosyVoice `emotion_instruct`.
