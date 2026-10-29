# Laptop audio-only demo

This checkout preserves the upstream LiveKit stage, host-moderated show graph,
per-persona context, PAD appraisal, hand-raise queue and speech/subtitle events.
For this local reproduction, text completion uses the user's existing Ark
`deepseek-v4-flash` subscription instead of a GPU-hosted Gemma server. The original
three distinct Fish Audio voices are now used for the local demo, with streaming
PCM playback through LiveKit. Piper remains an optional CPU-only fallback selected
explicitly by configuration. No avatar assets or GPU are needed.

## Start

From the repository root, after configuring the ignored `.env`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-agent-cpu.txt
bash deploy/run-local-agent.sh
```

In another terminal:

```bash
cd talkshow-web
pnpm install
pnpm build
pnpm start --hostname 127.0.0.1 --port 3001
```

Open `http://127.0.0.1:3001/rooms/talkshow-gefei-ready`, enter a name and join.
Type your point, click **Raise hand**, and wait for the host to grant the floor.
Then click **Send**. AI responses are generated live, spoken in three distinct
voices, and shown in the transcript. The host returns the floor after a bounded
panel round (four panel turns in this demo).

The first human turn is reserved automatically after welcome; later **Raise hand**
requests join the normal queue. Development is also supported with `pnpm dev`,
but source edits can reset the transcript/floor UI through Fast Refresh. Use the
production build for a stable demo, and rejoin a fresh room after changing code.

## Video's two-input example / audio check

```bash
SSL_CERT_FILE="$(.venv/bin/python -m certifi)" \
  .venv/bin/python -m eval.ghost_session.local_audio_demo \
  --room talkshow-gefei-audio-test \
  --script eval/ghost_session/scripts/pad_swing.yaml \
  --output logs/local-demo.wav
```

This uses the upstream video's same scripted human input, with live generated AI
responses and actual LiveKit audio capture. It verifies all three AI roles spoke,
that the floor returned after each input, and that the captured WAV is non-silent.
It does not launch a separate browser or depend on Playwright.

## Local-only configuration

Keep `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `VLLM_API_KEY`, and `FISH_API_KEY` in `.env` only.
Set `VLLM_BASE_URL` to the user's authorized OpenAI-compatible backend and
`VLLM_MODEL=deepseek-v4-flash`. For the laptop path:

```dotenv
TALKSHOW_TEXT_INPUT=1
NEXT_PUBLIC_TALKSHOW_TEXT_INPUT=1
TALKSHOW_AVATAR_ENABLED=0
NEXT_PUBLIC_DYSTREAM_ENABLED=0
TALKSHOW_PIPER_CUDA=0
TALKSHOW_TTS_ENGINE=fish
FISH_API_KEY=your-fish-api-key
FISH_MODEL=s2.1-pro-free
FISH_LATENCY=balanced
FISH_SAMPLE_RATE=44100
TALKSHOW_TURN_DETECTOR=vad
TALKSHOW_GEMMA_MM_WARM=0
PIPER_MODEL_PATH=models/piper/en_US-lessac-medium.onnx
PIPER_MODEL_PATH_COMMENTATOR=models/piper/en_US-ryan-medium.onnx
PIPER_MODEL_PATH_GUEST=models/piper/en_US-amy-medium.onnx
TALKSHOW_PANEL_MAX_TURNS=4
TALKSHOW_HOST_MODERATE_MAX_DEPTH=4
```

Fish uses the existing host/Ryan/Amy voice IDs from the earlier text-show demo.
Override them with `FISH_REFERENCE_ID_HOST`, `FISH_REFERENCE_ID_COMMENTATOR`, and
`FISH_REFERENCE_ID_GUEST`. Full sentences are sent to Fish; PCM audio chunks are
forwarded to LiveKit as they arrive. This is HTTP audio-output streaming, not
token-by-token text input. Interrupting closes the in-flight HTTP synthesis and
clears the LiveKit speech handle. Authorization and upstream failures are surfaced,
not silently replaced with Piper or a different Fish tier.

To use local Piper again, set `TALKSHOW_TTS_ENGINE=piper` and run
`.venv/bin/python deploy/download-local-piper.py` if the voices are not installed.
Restart the worker and join a fresh room after changing TTS configuration.

English only for this setup. Mic audio input is disabled: DeepSeek is text-only,
so this is **not** a reproduction of Gemma's direct audio understanding. This
checkout now also has Gefei's opt-in reactive floor-control path below. To
restore upstream input, disable reactive mode, remove the text-mode flags and
configure a working audio-capable Gemma server. Do not expose the development
server or unauthenticated token endpoint to the internet.

## Reactive floor-control demo (Gefei)

### AI-only presentation

Open `http://127.0.0.1:3001/rooms/gefei-observer-showcase?aiOnly=1`, click **Enter studio**,
then **Start demo**. Gefei Gu appears as a silent Human observer (letter avatar),
alongside the three AI-marked speakers Lessac, Ryan and Amy. There is no speaking human guest,
name entry, mic/camera publication, text input, hand raise, or human floor grant.
The topic is configuration, not a fabricated Human guest transcript line.
The original human-interruption mode remains available without `aiOnly=1`.
An audience token can start/stop the episode, but cannot inject human turns.
**Stop** cancels the episode as an operator action, not an in-world interruption.
Normal completion returns to Ready for another episode; host dialogue stays live.
Only a current-epoch granted AI rebuttal triggers the 8-second interruption banner.
The victim's caption keeps “Interrupted by Ryan”, and the takeover caption is
marked “Took the floor”; queued, stale, and normal turns do not trigger the banner.

The large topic banner shows the current `reactive_reset.topic`. Live captions
come from LiveKit's playback-synchronized text sink, not a new ASR service or a
fake typewriter over the full LLM draft. Each speech epoch updates one transcript
card; interruption closes that card and rejects late partial packets. Fish lacks
word timestamps, so word pacing is the SDK's audio-based estimate, not exact
forced alignment. Only fully played sentences still enter shared model history.

Local branch: `gefei-ai-interruption`. The old Sites / Fish interruption demo
is unchanged. This feature now uses three distinct Fish Audio voices and
the existing authorized DeepSeek backend, without avatars or microphone ASR.

The laptop start script defaults `TALKSHOW_REACTIVE=1`. For the original path:

```bash
TALKSHOW_REACTIVE=0 bash deploy/run-local-agent.sh
```

Join `http://127.0.0.1:3001/rooms/gefei-floor-control-demo` and wait for the
host to give you the floor. The **Floor control** panel has three short runs:

- **AI interrupt**: Amy takes a deliberately pro-AI-homework opening position.
  Ryan independently requests interrupt / queue / none using only played text.
  A grounded rebuttal can cut Amy. Ryan's already-generated reply takes over.
- **Queue only**: the same kind of live request cannot cut the speaker. Amy
  finishes, the stale queued draft is discarded, and Ryan generates a fresh
  response from the completed context.
- **Sequential**: Amy finishes before Ryan's ordinary LLM generation begins.
  No reactive-intent model calls run. This is a control condition, not a replay.

Each button starts a fresh working transcript with the preset homework topic
and opposing debate stances. AI dialogue and intent are live generated. A model
can choose not to interrupt; it is not a fixed timer or prerecorded exchange.
The bounded discussion now plays Amy → Ryan → Amy → Lessac's follow-up → Ryan's
proposal → Amy's final response → Lessac's contextual wrap (seven speeches).
Only pacing/order and the opening stance are preset, not dialogue or the decision
to interrupt. After the first exchange, AI requests wait rather than repeatedly
cutting replies. Queued drafts are never reused after context advances; follow-up
speech is generated from the actual played history. Human priority remains active,
including while a follow-up is being generated. The generated wrap ends the beat
and silently restores human text input; no extra canned "Floor's yours" follows.
Demo-only AI queue entries are cleared on completion/stop. Full telemetry remains in
the ignored `logs/session-*.jsonl` files even when the UI transcript resets.

**Interrupt** during live character speech or a demo generation gap grants the human priority. The audio
handle stops, pending role requests are invalidated/cancelled, and text input
unlocks. Send your point to hear the host respond. This is button-triggered
interruption, not microphone barge-in. During canned procedural lines, or when
no reactive line or demo generation is active, the interruption button is disabled.

### Implementation and policy

- `agent/floor/reactive_policy.py`: pure deterministic rules. Human priority 100,
  grounded rebuttal/correction 50, ordinary addition 10. AI minimum speaking
  time defaults to 3 s, global AI-cut cooldown to 15 s. Human cuts bypass those.
- `agent/multi_agent/reactive_session.py`: one speech handle at a time, one
  concurrent intent call per eligible listening panelist after the first fully
  played sentence. At most one AI cut per reactive chain. The moderator does
  not compete as a reactive panelist in this MVP.
- `agent/multi_agent/reactive_demo_plan.py`: two short exchanges, a moderator
  follow-up and a fresh contextual ending. A rules-based seven-speech budget
  prevents infinite rounds; the LLM generates each persona's own reply.
- A model cannot assign its own role, epoch or source. JSON/enums/lengths are
  validated. Rebuttal evidence must quote text already played. This checks
  grounding and rule eligibility, not the factual correctness of the argument.
- Monotonic turn epochs reject old completions even when the transcript cap
  holds its length constant. Network, timeout and parse failures fail closed.
- Only fully played sentences enter shared history. A partly audible interrupted
  sentence is conservatively omitted; this is sentence-level accounting, not
  word-accurate audio alignment. The full unplayed draft stays out of history.
- The graph's normal prepare/record path is bypassed for reactive substantive
  lines, so PAD appraisals also see only the committed played prefix. Ordinary
  mode and its original parallel TTS/PAD graph remain available.
- `reactive_decision` events expose the request source, decision, rule and quote.
  **Export** opens copyable JSONL plus a file-download option for the retained
  client decision history (up to 100 records). Use **Copy JSONL** in an embedded
  browser that does not support file downloads.
  Server JSONL includes complete request, cancellation and sentence telemetry.

- Listeners re-decide after every fully played sentence, with at most one poll in
  flight (`TALKSHOW_REACTIVE_LISTEN=first` restores the single first-sentence poll).
- The moderator also listens and may make a priority-80 "moderation" cut when a
  panelist clearly leaves the topic or attacks someone; it bypasses the AI cooldown
  and chain cap, still waits for the minimum speaking time, and hands the floor back
  to the speaker it cut (`TALKSHOW_REACTIVE_HOST_CUT=0` disables it).
- Quotes are matched as word sequences, ignoring case, punctuation and quotation marks.

Useful optional configuration:

```dotenv
TALKSHOW_REACTIVE_MIN_SPEECH_S=3
TALKSHOW_REACTIVE_COOLDOWN_S=15
TALKSHOW_REACTIVE_POLL_TIMEOUT_S=12
TALKSHOW_REACTIVE_LISTEN=every
TALKSHOW_REACTIVE_HOST_CUT=1
# OpenAI gpt-5+/o-series: reasoning is off unless set (low | medium | high)
TALKSHOW_LLM_REASONING_EFFORT=none
```

Offline evaluation of these decisions (planted overclaims, factual errors and
off-topic drifts, with arms replayed from cached answers): `eval/floor_eval/`,
results in `eval/floor_eval/results/README.md`.

### Acceptance and reproducibility

```bash
.venv/bin/python -m unittest discover -s tests/show_graph -q
.venv/bin/python -m unittest discover -s tests/session -q
.venv/bin/python -m unittest discover -s tests/emotion -q
cd talkshow-web
pnpm test
pnpm exec tsc --noEmit
pnpm build
```

From the repository root, with the local worker running:

```bash
SSL_CERT_FILE="$(.venv/bin/python -m certifi)" \
  .venv/bin/python -m eval.ghost_session.reactive_acceptance
```

This creates isolated rooms for live AI cuts (twice), queue-only, sequential,
and human cut followed by actual text input and a host reply. It captures real
received LiveKit PCM and checks non-silence, decisions, committed prefixes and
omitted tails. Outputs: `logs/reactive-acceptance/summary.json`, per-room UI-event
captures and WAV audio. A final regression set uses `logs/reactive-final-acceptance/`.
The exploratory first pilot chose queue rather than cut and is preserved in
`logs/session-1791176448.jsonl`; the clearer opposing-stance case follows it.

`stop_ack_ms` is server speech-handle completion after the controller grants a
cut. It is **not** the browser's acoustic stop latency. LiveKit jitter buffering,
PAD appraisal and next-voice startup still affect the perceived handoff. Total
demo duration is not a fair speed benchmark because interrupted speech is
shorter and generated response lengths vary. The RTC SDK can emit destructor
warnings when the automated recorder process exits; evaluate runtime events
and captured audio separately from these cleanup warnings.
