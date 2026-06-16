# VRM avatar assets (local dev)

Binary `.vrm` files are **not committed** (see `.gitignore`). Fetch once:

```bash
cd talkshow-web
bash scripts/fetch-vrm-samples.sh
```

## Model files → panel roles

Files are named after the **source model**, not Piper voice / persona names.

| File | Role (persona) | Source model |
|------|----------------|--------------|
| `seed-san.vrm` | Host (Lessac) | [Seed-san](https://github.com/vrm-c/vrm-specification/tree/master/samples/Seed-san) |
| `seed-san.vrm` or `vrm1-twist-spec.vrm` | Commentator (Ryan) | Same as old guest slot (Seed-san, or vrm-spec VRM1 if hashes differ) |
| `vrm1-twist-sample.vrm` | Guest (Amy) | [VRM1_Constraint_Twist_Sample](https://github.com/pixiv/three-vrm/tree/dev/packages/three-vrm/examples/models) (pixiv / three-vrm copy) |

Legacy names `lessac-head.vrm`, `ryan-head.vrm`, `amy-head.vrm` are removed by the fetch script.

Original downloads are kept under `sources/` for attribution.

## License

Both practical samples use **[VRM Public License 1.0](https://vrm.dev/en/licenses/1.0/)**:

- **Seed-san** — VirtualCast, Inc. ([sample README](https://github.com/vrm-c/vrm-specification/blob/master/samples/Seed-san/README.md))
- **VRM1_Constraint_Twist_Sample** — pixiv Inc. ([sample README](https://github.com/vrm-c/vrm-specification/blob/master/samples/VRM1_Constraint_Twist_Sample/README.md))

Use for **Avatar Use** in this talk-show demo. Replace with custom VRoid exports before public release if you need distinct branded characters.
