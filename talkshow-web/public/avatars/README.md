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

Speech video is LiveKit only; idle is this static loop.

## VRM (optional / legacy)

Binary `.vrm` files stay under this directory (gitignored). Fetch:

```bash
cd talkshow-web
bash scripts/fetch-vrm-samples.sh
```

| File | Role | Source |
|------|------|--------|
| `seed-san.vrm` | Host | [Seed-san](https://github.com/vrm-c/vrm-specification/tree/master/samples/Seed-san) |
| `vrm1-twist-sample.vrm` | Guest | [three-vrm sample](https://github.com/pixiv/three-vrm/tree/dev/packages/three-vrm/examples/models) |

Originals under `sources/`. License: [VRM Public License 1.0](https://vrm.dev/en/licenses/1.0/).
