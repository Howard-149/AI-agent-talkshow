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
