#!/usr/bin/env python3
"""Listening test, response shift R: does an edit to one human turn change what the panel says?

Input: ghost replay sessions (eval/data/minimal_pairs/make_pairs.py conditions). Every session replays the same saved
show state (prefix), then injects turn 2 under one condition (orig | paraphrase | negate | number | entity | addressee |
persona_swap), K resamples each.

Stages, in order (embed and logprob need a GPU; deploy/slurm-eval-listen.sh):
    collect   session logs -> <out>/lines.jsonl (panel lines after the edit) + sessions.json
    embed     lines.jsonl -> <out>/emb.npy (vLLM pooling model)
    logprob   lines.jsonl -> <out>/lp.jsonl (open scorer: lp(y | own context) and lp(y | context with turn 2 swapped to
              the orig text, or the persona swapped back))
    report    -> <out>/responsiveness.json + printed table

Measures, per condition c against orig, on panel line position j after the edit:
    D_edit(c)  mean cosine distance between orig samples and c samples
    D_noise    mean cosine distance between distinct orig samples
    R(c)       sum over topics of D_edit / sum over topics of D_noise. 95% CI: bootstrap over topics (all K samples of a
               topic stay together). R near 1 = the edit moves the reply no more than resampling does (paraphrase).
    KL_lp(c)   symmetrised per-token log-likelihood gap of the replies between the two contexts (0 for orig resamples)
    landing    share of samples whose first panel line comes from the addressed panelist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANEL_ROLES = ("guest", "commentator")
# Room = ghost-<slug>__<cond>__k<k>[__v<arm>]-<job>-<epoch> (script names from make_pairs.py conditions).
ROOM_RE = re.compile(r"^ghost-(?P<slug>.+?)__(?P<cond>[a-z_]+)__k(?P<k>\d+)(?:__v(?P<arm>[A-Za-z0-9]+))?-\d+-\d+$")
# Other topic sets (e.g. the public test set built into eval/data/public_set/build/, git-ignored raw text):
# point TALKSHOW_PAIRS / TALKSHOW_PREFIX_DIR at them.
PAIRS = Path(os.environ.get("TALKSHOW_PAIRS", ROOT / "eval" / "data" / "minimal_pairs" / "pairs.yaml"))
PREFIX_DIR = Path(os.environ.get("TALKSHOW_PREFIX_DIR", ROOT / "eval" / "data" / "minimal_pairs" / "prefix"))
ADDRESSED = {"orig": "commentator", "addressee": "guest"}  # pairs.yaml: orig → Ryan, addressee → Amy


def _hash_lines(lines: list[dict]) -> str:
    blob = json.dumps([[l["role_id"], l["text"]] for l in lines], ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- collect


def cmd_collect(args: argparse.Namespace) -> None:
    import yaml

    from eval.log_parse import load_events

    pairs = {t["slug"]: t for t in yaml.safe_load(PAIRS.read_text())["topics"]}
    prefix_hash = {
        p.stem: _hash_lines(json.loads(p.read_text())["lines"]) for p in PREFIX_DIR.glob("*.json")
    }
    rows, sessions, bad = [], [], 0
    seen_cells: set[tuple[str, str, int, str]] = set()
    for log_dir in args.logs:
        for path in sorted(Path(log_dir).glob("session-*.jsonl")):
            events = load_events(path)
            room = next((e.get("room", "") for e in events if e["event"] == "session_start"), "")
            m = ROOM_RE.match(room)
            if not m:
                continue
            slug, cond, k = m["slug"], m["cond"], int(m["k"])
            arm = m["arm"] or ""
            if args.arm and arm != args.arm:
                continue
            snaps = [e for e in events if e["event"] == "history_snapshot"]
            done = [i for i, e in enumerate(events) if e["event"] == "ghost_human_done"]
            sess = {"slug": slug, "cond": cond, "k": k, "log": path.name, "ok": False}
            sessions.append(sess)
            if not snaps or not done:
                sess["why"] = "no snapshot/turn"
                bad += 1
                continue
            sess["prefix_match"] = _hash_lines(snaps[0]["lines"]) == prefix_hash.get(slug)
            heard = events[done[0]].get("heard", "")
            panel = []
            for e in events[done[0] + 1 :]:
                # Only the beat the edited turn started; a later idle-moderation beat is not its response.
                if e["event"] in ("ghost_human_start", "gemma_stt_start", "show_beat_done"):
                    break
                if e["event"] == "line_start" and e.get("role") in ("host", *PANEL_ROLES) and e.get("text"):
                    panel.append({"role": e["role"], "step": e.get("step", ""), "text": e["text"]})
            sess["ok"] = bool(sess["prefix_match"]) and any(p["role"] in PANEL_ROLES for p in panel)
            if not sess["ok"]:
                sess["why"] = "prefix mismatch" if not sess["prefix_match"] else "no panel line"
                bad += 1
                continue
            if (slug, cond, k, arm) in seen_cells:
                # Re-runs of a cell (fill / requeue): keep the first usable session only.
                sess["ok"] = False
                sess["why"] = "duplicate cell"
                continue
            seen_cells.add((slug, cond, k, arm))
            j = 0
            for idx, line in enumerate(panel):
                if line["role"] not in PANEL_ROLES:
                    continue
                j += 1
                rows.append(
                    {
                        "slug": slug,
                        "topic": pairs[slug]["topic"],
                        "cond": cond,
                        "k": k,
                        "pos": j,
                        "role": line["role"],
                        "text": line["text"],
                        "heard": heard,
                        # Lines spoken since the edited turn, before this one (incl. host).
                        "between": panel[:idx],
                        "log": path.name,
                    }
                )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "lines.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "sessions.json").write_text(json.dumps(sessions, indent=1))
    n_ok = sum(s["ok"] for s in sessions)
    cells = defaultdict(int)
    for s in sessions:
        if s["ok"]:
            cells[(s["slug"], s["cond"])] += 1
    print(f"sessions={len(sessions)} ok={n_ok} bad={bad} lines={len(rows)} → {out}/lines.jsonl")
    print(f"prefix identical in {sum(bool(s.get('prefix_match')) for s in sessions)}/{len(sessions)} sessions")
    short = {f"{a}/{b}": n for (a, b), n in cells.items() if n < args.k}
    if short:
        print(f"cells with < {args.k} ok sessions: {short}")


# --------------------------------------------------------------------------- embed


def cmd_embed(args: argparse.Namespace) -> None:
    import numpy as np
    from vllm import LLM

    out = Path(args.out)
    rows = [json.loads(l) for l in (out / "lines.jsonl").read_text().splitlines()]
    llm = LLM(model=args.embed_model, runner="pooling", gpu_memory_utilization=0.5)
    res = llm.embed([r["text"] for r in rows])
    emb = np.array([r.outputs.embedding for r in res], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(out / "emb.npy", emb)
    print(f"embedded {len(rows)} lines with {args.embed_model} → {out}/emb.npy")


# --------------------------------------------------------------------------- logprob


def _transcript_prompt(topic: str, history: list[dict], name: str) -> str:
    from eval.lp_score import HUMAN_LABEL, load_persona

    names = {r: load_persona(r)[0] for r in ("host", *PANEL_ROLES)}
    body = "\n".join(
        f"{HUMAN_LABEL if h['role'] == 'human' else names[h['role']]}: {h['text']}" for h in history
    )
    return (
        f"Live talk-show panel. Topic: {topic}\n\nTranscript so far:\n{body}\n\n"
        f"Your turn, {name}. Reply with your spoken line only."
    )


def cmd_logprob(args: argparse.Namespace) -> None:
    import yaml

    from eval.lp_score import load_persona, score

    out = Path(args.out)
    rows = [json.loads(l) for l in (out / "lines.jsonl").read_text().splitlines()]
    pairs = {t["slug"]: t for t in yaml.safe_load(PAIRS.read_text())["topics"]}
    personas = {r: load_persona(r) for r in PANEL_ROLES}
    prefixes = {
        p.stem: [
            {"role": l["role_id"], "text": l["text"]}
            for l in json.loads(p.read_text())["lines"]
            if not l.get("procedural")
        ]
        for p in PREFIX_DIR.glob("*.json")
    }
    requests, keys = [], []
    for i, r in enumerate(rows):
        name, persona = personas[r["role"]]
        other_persona = personas[next(x for x in PANEL_ROLES if x != r["role"])][1]
        orig_text = pairs[r["slug"]]["turn2"]["orig"]

        def ctx(human_text: str) -> list[dict]:
            return prefixes[r["slug"]] + [{"role": "human", "text": human_text}] + [
                {"role": b["role"], "text": b["text"]} for b in r["between"]
            ]

        own = (persona, _transcript_prompt(r["topic"], ctx(r["heard"]), name))
        if r["cond"] == "persona_swap":
            # The intervention is the persona: score under the swapped (actually used) one vs the real one.
            own = (other_persona, own[1])
            alt = (persona, own[1])
        elif r["cond"] == "orig":
            alt = None  # filled per paired condition at report time via the edit texts
        else:
            alt = (persona, _transcript_prompt(r["topic"], ctx(orig_text), name))
        conds = {"own": own}
        if alt is not None:
            conds["alt"] = alt
        if r["cond"] == "orig":
            # orig lines scored under every edited turn-2 text (and the swapped persona).
            for c, text in pairs[r["slug"]]["turn2"].items():
                if c != "orig":
                    conds[f"alt:{c}"] = (persona, _transcript_prompt(r["topic"], ctx(text), name))
            conds["alt:persona_swap"] = (other_persona, own[1])
        for c, (system, user) in conds.items():
            requests.append((None, c, system, user, r["text"]))
            keys.append((i, c))
    print(f"{len(rows)} lines, {len(requests)} scoring requests, model={args.lp_model}", file=sys.stderr)
    scored = score(requests, args.lp_model, args.max_model_len)
    lp: dict[int, dict] = defaultdict(dict)
    for (i, c), (total, n) in zip(keys, scored):
        lp[i][c] = total / max(n, 1)
    with (out / args.lp_name).open("w") as f:
        for i in range(len(rows)):
            f.write(json.dumps(lp[i]) + "\n")
    print(f"wrote {out}/{args.lp_name}")


# --------------------------------------------------------------------------- report


def _ci(xs: list[float]) -> tuple[float, float]:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan"), float("nan")
    return xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]


def cmd_report(args: argparse.Namespace) -> None:
    import numpy as np

    out = Path(args.out)
    rows = [json.loads(l) for l in (out / "lines.jsonl").read_text().splitlines()]
    emb = np.load(out / "emb.npy") if (out / "emb.npy").is_file() else None
    lp = (
        [json.loads(l) for l in (out / args.lp_name).read_text().splitlines()]
        if (out / args.lp_name).is_file()
        else None
    )
    # Stage outputs are row-aligned with lines.jsonl; a re-collect makes them stale.
    if emb is not None and len(emb) != len(rows):
        print(f"WARN emb.npy has {len(emb)} rows, lines.jsonl {len(rows)} — re-run embed; ignoring", file=sys.stderr)
        emb = None
    if lp is not None and len(lp) != len(rows):
        print(f"WARN lp.jsonl has {len(lp)} rows, lines.jsonl {len(rows)} — re-run logprob; ignoring", file=sys.stderr)
        lp = None
    rng = random.Random(args.seed)
    conds = sorted({r["cond"] for r in rows} - {"orig"})
    slugs = sorted({r["slug"] for r in rows})
    # idx[pos][slug][cond] -> row indices (one per resample that reached that position)
    idx: dict[int, dict[str, dict[str, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, r in enumerate(rows):
        idx[r["pos"]][r["slug"]][r["cond"]].append(i)
    # Per-panelist: each session's first post-edit line by that panelist.
    seen_role: set[tuple[str, str]] = set()
    for i, r in sorted(enumerate(rows), key=lambda x: (x[1]["log"], x[1]["pos"])):
        if (r["log"], r["role"]) not in seen_role:
            seen_role.add((r["log"], r["role"]))
            idx[f"{r['role']}_first"][r["slug"]][r["cond"]].append(i)  # type: ignore[index]

    def dist(a: list[int], b: list[int], same: bool) -> tuple[float, int]:
        if emb is None or not a or not b:
            return 0.0, 0
        sims = emb[a] @ emb[b].T
        d = np.clip(1.0 - sims, 0.0, None)
        if same:
            n = len(a)
            mask = ~np.eye(n, dtype=bool)
            return float(d[mask].sum()), int(mask.sum())
        return float(d.sum()), int(d.size)

    def ratio(pos_set: list, cond: str, sample: bool) -> float:
        num = den = 0.0
        topics = [rng.choice(slugs) for _ in slugs] if sample else slugs
        for s in topics:
            for pos in pos_set:
                cell = idx[pos].get(s, {})
                o, e = cell.get("orig", []), cell.get(cond, [])
                if len(o) < 2 or not e:
                    continue
                ds, ns = dist(o, e, False)
                dn, nn = dist(o, o, True)
                if ns and nn:
                    num += ds / ns
                    den += dn / nn
        return num / den if den else float("nan")

    report: dict = {"n_lines": len(rows), "topics": slugs, "positions": {}}
    sess_path = out / "sessions.json"
    if sess_path.is_file():
        sessions = json.loads(sess_path.read_text())
        counts: dict = defaultdict(lambda: {"ok": 0, "bad": 0})
        for sess in sessions:
            if sess.get("why") == "duplicate cell":
                continue
            counts[sess["cond"]]["ok" if sess["ok"] else "bad"] += 1
        report["sessions"] = dict(counts)
    report["lines_per_cond"] = {
        c: sum(r["cond"] == c for r in rows) for c in ["orig", *conds]
    }
    positions = {
        "pos1": [1],
        "pos1-3": [1, 2, 3],
        "guest_first": ["guest_first"],
        "commentator_first": ["commentator_first"],
    }
    if emb is not None:
        for label, pos_set in positions.items():
            report["positions"][label] = {}
            for c in conds:
                point = ratio(pos_set, c, False)
                boots = [ratio(pos_set, c, True) for _ in range(args.boot)]
                lo, hi = _ci(boots)
                report["positions"][label][c] = {"R": round(point, 3), "ci95": [round(lo, 3), round(hi, 3)]}

    def kl_table(role_filter: str | None, pos1_only: bool = True) -> dict:
        kl: dict[str, dict] = {}
        for c in conds:
            per_topic = []
            for s in slugs:
                def keep(r: dict, cond: str) -> bool:
                    return (r["slug"] == s and r["cond"] == cond
                            and (role_filter is None or r["role"] == role_filter)
                            # pos1 = the line that answers the edited turn (model-only: every line)
                            and (not pos1_only or r["pos"] == 1))
                e_rows = [i for i, r in enumerate(rows) if keep(r, c)]
                o_rows = [i for i, r in enumerate(rows) if keep(r, "orig")]
                if not e_rows or not o_rows:
                    continue
                a = [lp[i]["own"] - lp[i]["alt"] for i in e_rows if "alt" in lp[i]]
                b = [lp[i]["own"] - lp[i][f"alt:{c}"] for i in o_rows if f"alt:{c}" in lp[i]]
                if a and b:
                    per_topic.append(0.5 * (sum(a) / len(a) + sum(b) / len(b)))
            if per_topic:
                boots = [
                    sum(rng.choice(per_topic) for _ in per_topic) / len(per_topic) for _ in range(args.boot)
                ]
                lo, hi = _ci(boots)
                kl[c] = {
                    "KL_lp": round(sum(per_topic) / len(per_topic), 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "topics": len(per_topic),
                }
        return kl

    if lp is not None:
        report["KL_lp"] = kl_table(None)  # pos1
        report["KL_lp_allpos"] = kl_table(None, pos1_only=False)
        report["KL_lp_by_role"] = {role: kl_table(role) for role in PANEL_ROLES}

    landing = {}
    for c, addressed in ADDRESSED.items():
        firsts = [r for r in rows if r["cond"] == c and r["pos"] == 1]
        if firsts:
            landing[c] = {
                "addressed": addressed,
                "first_line_by_addressed": round(sum(r["role"] == addressed for r in firsts) / len(firsts), 3),
                "n": len(firsts),
            }
    report["landing"] = landing
    report_name = "responsiveness.json" if args.lp_name == "lp.jsonl" else f"responsiveness_{Path(args.lp_name).stem}.json"
    (out / report_name).write_text(json.dumps(report, indent=1))

    print(f"lines={len(rows)} topics={len(slugs)}")
    for label, table in report["positions"].items():
        print(f"\nembedding R ({label}):")
        for c, v in table.items():
            print(f"  {c:13s} R={v['R']:.3f}  95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    if report.get("KL_lp"):
        print("\nscorer KL_lp (nats/token, pos1):")
        for c, v in report["KL_lp"].items():
            print(f"  {c:13s} {v['KL_lp']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]  topics={v['topics']}")
    for role, table in (report.get("KL_lp_by_role") or {}).items():
        print(f"\nscorer KL_lp, speaker={role}:")
        for c, v in table.items():
            print(f"  {c:13s} {v['KL_lp']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]  topics={v['topics']}")
    print(f"\nlanding: {landing}")
    print(f"→ {out}/{report_name}")


STAGES_HELP = "stages run in order: collect (CPU), embed (GPU), logprob (GPU), report (CPU)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=STAGES_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "resp"), help="results dir (one per measured run)")
    ap.add_argument("--lp-name", default="lp.jsonl", help="logprob file (a second scorer writes e.g. lp_llama.jsonl)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect", help="session logs -> lines.jsonl (CPU)")
    p.add_argument("logs", nargs="+", help="dirs holding session-*.jsonl")
    p.add_argument("-k", type=int, default=5, help="expected resamples per cell (only for the warning)")
    p.add_argument("--arm", default="", help="keep only sessions of this arm (room suffix __v<arm>)")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("embed", help="lines.jsonl -> emb.npy (GPU)")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.set_defaults(fn=cmd_embed)
    p = sub.add_parser("logprob", help="lines.jsonl -> lp.jsonl (GPU)")
    p.add_argument("--lp-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.set_defaults(fn=cmd_logprob)
    p = sub.add_parser("report", help="R, KL_lp and landing -> responsiveness.json (CPU)")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
