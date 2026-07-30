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
const IDLE_DEBOUNCE_MS = 280;

/** Agent track energy — drives VRM mouth only (not transcript ordering). */
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

/**
 * Transcript + highlight follow agent `role_active` / `transcript` events (speaking timing).
 * Lip sync still follows the WebRTC audio track.
 */
export function usePlayoutSyncedUi(agentParticipant: Participant | undefined) {
  const lipSyncSpeaking = useAgentTrackSpeaking(agentParticipant);
  const [signaledRole, setSignaledRole] = useState<PanelRole | null>(null);
  const [lines, setLines] = useState<SyncedTranscriptLine[]>([]);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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
      const trimmed = text.trim();
      if (!trimmed) return;
      appendLine(role, speaker, trimmed);
    },
    [appendLine],
  );

  const lipSyncActive = useMemo(
    () => !!(signaledRole && signaledRole !== 'human' && lipSyncSpeaking),
    [signaledRole, lipSyncSpeaking],
  );

  return { activeRole, lipSyncActive, lines, onRoleActive, onRoleIdle, pushLine };
}
