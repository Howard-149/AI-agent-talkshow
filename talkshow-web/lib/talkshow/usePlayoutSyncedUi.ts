'use client';

import { ParticipantEvent, type Participant, type RemoteParticipant } from 'livekit-client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { parseAgentMetadata, type PanelRole } from '@/lib/talkshow/roles';

export type SyncedTranscriptLine = {
  id: string;
  speaker: string;
  role: string;
  text: string;
  ts: number;
};

/** Remote talkshow agent participant (single RTC bot). */
export function findAgentParticipant(
  remotes: RemoteParticipant[],
): RemoteParticipant | undefined {
  return remotes.find((p) => parseAgentMetadata(p.metadata)?.talkshowAgent);
}

/** Hold lip-sync through brief dips between TTS sentences in one utterance. */
const SPEAKING_HOLD_MS = 450;
/** Defer clearing speaker highlight between back-to-back lines. */
const IDLE_DEBOUNCE_MS = 500;

/** Agent track energy — drives VRM mouth only (not transcript/highlight gating). */
function useAgentTrackSpeaking(agentParticipant: Participant | undefined): boolean {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const holdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!agentParticipant) {
      setIsSpeaking(false);
      return;
    }
    const apply = (raw: boolean) => {
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current);
        holdTimerRef.current = null;
      }
      if (raw) {
        setIsSpeaking(true);
        return;
      }
      holdTimerRef.current = setTimeout(() => {
        setIsSpeaking(false);
        holdTimerRef.current = null;
      }, SPEAKING_HOLD_MS);
    };
    const sync = () => apply(agentParticipant.isSpeaking);
    sync();
    agentParticipant.on(ParticipantEvent.IsSpeakingChanged, sync);
    return () => {
      agentParticipant.off(ParticipantEvent.IsSpeakingChanged, sync);
      if (holdTimerRef.current) clearTimeout(holdTimerRef.current);
    };
  }, [agentParticipant]);

  return isSpeaking;
}

/** Split panel line into speakable sentences for progressive reveal. */
function splitSpeakSentences(text: string): string[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const parts = trimmed.match(/[^.!?]+[.!?]+|[^.!?]+$/g);
  return parts?.map((s) => s.trim()).filter(Boolean) ?? [trimmed];
}

/**
 * UI events are timed on the agent when playout starts (speaking state).
 * Lip sync still follows the WebRTC audio track.
 */
export function usePlayoutSyncedUi(agentParticipant: Participant | undefined) {
  const lipSyncSpeaking = useAgentTrackSpeaking(agentParticipant);
  const [signaledRole, setSignaledRole] = useState<PanelRole | null>(null);
  const [lines, setLines] = useState<SyncedTranscriptLine[]>([]);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Progressive reveal for multi-sentence lines (StreamAdapter sentence chunks).
  const revealQueueRef = useRef<
    Array<{ role: string; speaker: string; sentences: string[]; idx: number }>
  >([]);
  const wasSpeakingRef = useRef(false);

  const appendLine = useCallback((role: string, speaker: string, text: string) => {
    const line: SyncedTranscriptLine = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      role,
      speaker,
      text,
      ts: Date.now(),
    };
    setLines((prev) => [...prev, line]);
  }, []);

  const flushRevealSentence = useCallback(() => {
    const head = revealQueueRef.current[0];
    if (!head || head.idx >= head.sentences.length) return;
    appendLine(head.role, head.speaker, head.sentences[head.idx]!);
    head.idx += 1;
    if (head.idx >= head.sentences.length) {
      revealQueueRef.current.shift();
    }
  }, [appendLine]);

  useEffect(() => {
    const rising = lipSyncSpeaking && !wasSpeakingRef.current;
    wasSpeakingRef.current = lipSyncSpeaking;
    if (rising) {
      flushRevealSentence();
    }
  }, [lipSyncSpeaking, flushRevealSentence]);

  const activeRole = signaledRole;

  const onRoleActive = useCallback((role: PanelRole) => {
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
    setSignaledRole(role);
  }, []);

  const onRoleIdle = useCallback(() => {
    if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    idleTimerRef.current = setTimeout(() => {
      idleTimerRef.current = null;
      setSignaledRole(null);
    }, IDLE_DEBOUNCE_MS);
  }, []);

  const pushLine = useCallback(
    (role: string, speaker: string, text: string) => {
      if (role === 'human') {
        appendLine(role, speaker, text);
        return;
      }
      const sentences = splitSpeakSentences(text);
      if (sentences.length <= 1) {
        appendLine(role, speaker, text);
        return;
      }
      revealQueueRef.current.push({ role, speaker, sentences, idx: 0 });
      // First sentence often aligns with the speaking edge that triggered this event.
      if (lipSyncSpeaking) {
        flushRevealSentence();
      }
    },
    [appendLine, flushRevealSentence, lipSyncSpeaking],
  );

  const lipSyncActive = useMemo(
    () => !!(signaledRole && signaledRole !== 'human' && lipSyncSpeaking),
    [signaledRole, lipSyncSpeaking],
  );

  return { activeRole, lipSyncActive, lines, onRoleActive, onRoleIdle, pushLine };
}
