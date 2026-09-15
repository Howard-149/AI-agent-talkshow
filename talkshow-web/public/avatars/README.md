# Avatar assets (laptop `talkshow-web`)

## DyStream idle + portraits — single source

**Canonical files:** repo-root [`avatar/assets/`](../../avatar/assets/) (`loops/`, `portraits/`).

This folder’s `loops` and `portraits` are **symlinks** into that tree. Do not copy MP4/PNG here.

```bash
# After bake on Babel (or locally), refresh the repo assets once:
scp babel:~/AI-agent-talkshow/avatar/assets/loops/*-idle.mp4 avatar/assets/loops/
scp babel:~/AI-agent-talkshow/avatar/assets/portraits/*.png avatar/assets/portraits/
# Frontend picks them up via the symlinks — no second copy under talkshow-web.
```

Browser URLs stay `/avatars/loops/...` and `/avatars/portraits/...` (Next `public/`).

Speech video is LiveKit only (agent `talkshow-avatar` track when `NEXT_PUBLIC_DYSTREAM_ENABLED=1`); idle is the static loop MP4 per role.
