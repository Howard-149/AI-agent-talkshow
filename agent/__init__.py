"""Talk-show LiveKit agent worker.

Package map (find features by folder):

- ``agent.floor`` — hand-raise queue, host moderation, turn controller
- ``agent.panel`` — panelist prompts, speech, roster
- ``agent.session`` — LiveKit session, handoff, speak_panel_line
- ``agent.show`` — transcript, scene context, TalkShowData
- ``agent.ui`` — data-channel UI/control events
- ``agent.locale`` — viewer locales + localization
- ``agent.emotion`` — role emotion tags + CosyVoice instruct
- ``agent.telemetry`` — JSONL turn/latency logs
- ``agent.adapters`` — Gemma / Piper / CosyVoice / DyStream adapters
- ``agent.agents`` — LiveKit Agent role subclasses
"""
