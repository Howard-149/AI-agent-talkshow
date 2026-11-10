# CosyVoice clone prompt WAVs (not committed — bake locally / on Babel)

Bake from Piper voices (one wav per role — **language-agnostic** for instruct2):

```bash
bash deploy/bake-cosyvoice-prompts-from-piper.sh
```

Expected files (basenames in `config/personas/*.yaml` → `tts.cosyvoice.prompt_wav`):

| File | Role | Typical Piper source |
|------|------|----------------------|
| `en_host.wav` | host (all locales) | Lessac |
| `en_guest.wav` | guest (all locales) | Amy |
| `en_commentator.wav` | commentator (all locales) | Ryan |

`zh_*.wav` is unused when personas point at `en_*.wav`.

Resolved as `<repo>/cosyvoice-prompts/<basename>`.  
`.env` only needs `TALKSHOW_TTS_ENGINE=cosyvoice`, `COSYVOICE_MODE=instruct2`, sidecar URL / model dir.

## Per-emotion refs (optional)

`COSYVOICE_EMOTION_REFS=1` picks the reference by the line's emotion from
`msp/<role>/<Emotion>.wav` — one MSP-Podcast speaker per role, clips listed in
`config/msp_emotion_refs.yaml`. Labels without a clip use that speaker's `Neutral.wav`.
The audio is not in git (MSP-Podcast licence: academic use, no redistribution). On Babel,
point at the shared baked copy (readable by everyone, compute nodes only):

    COSYVOICE_EMOTION_REFS=1
    COSYVOICE_EMOTION_REFS_DIR=/data/user_data/hsuanhal/msp_emotion_refs

or bake your own where MSP-Podcast is mounted: `python deploy/bake_msp_emotion_refs.py`.
