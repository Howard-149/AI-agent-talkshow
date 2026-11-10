'use client';

/**
 * Syncs transcript lines, speaker highlight, and lip-sync UI to agent audio playout timing.
 *
 * Agent lines stream word by word from the agent's synced `lk.transcription` text
 * streams (paced to the real TTS audio). The `transcript` UI event that opens a line is
 * published before its audio on the same reliable data channel, so every stream that
 * arrives belongs to the most recently opened agent line.
 */
import { useRoomContext } from '@livekit/components-react';
import {
  ParticipantEvent,
  type Participant,
  type RemoteParticipant,
  type TextStreamReader,
} from 'livekit-client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { parseAgentMetadata, type PanelRole } from '@/lib/talkshow/roles';
import { joinSegments, type TranscriptBackchannel } from '@/lib/talkshow/transcriptParts';

export type { TranscriptBackchannel } from '@/lib/talkshow/transcriptParts';

export type SyncedTranscriptLine = {
  id: string;
  speaker: string;
  role: string;
  /** Text shown so far (synced words, or the fallback). */
  text: string;
  ts: number;
  /** Words still arriving from the synced transcription stream. */
  streaming?: boolean;
  /** Name of the panelist whose approved interrupt cut this line. */
  interruptedBy?: string;
  /** This line is an interrupter taking the floor. */
  tookFloor?: boolean;
  backchannels: TranscriptBackchannel[];
};

export type InterruptNotice = { id: number; name: string; byName: string };

type LineOptions = {
  /** Reveal the words from the synced transcription stream (agent lines). */
  stream?: boolean;
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
/** Agent synced transcription topic (livekit-agents room output). */
const TRANSCRIPTION_TOPIC = 'lk.transcription';
/** No synced words this long after a line opened: show its full text instead. */
const STREAM_FALLBACK_MS = 4000;
/** How long the "X interrupts Y" notice stays up. */
const INTERRUPT_NOTICE_MS = 6000;
/** An interrupt marks the interrupter's next line if it starts within this window. */
const TOOK_FLOOR_WINDOW_MS = 20000;
/** A settled line still takes late synced words for this long. */
const SEGMENT_RETAIN_MS = 10000;
/** Extra highlight time after a backchannel's audio ends. */
const BACKCHANNEL_TAIL_MS = 300;

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

/** Agent track energy — drives speaking glow / lip sync hint (not transcript ordering). */
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
  const room = useRoomContext();
  const lipSyncSpeaking = useAgentTrackSpeaking(agentParticipant);
  const [signaledRole, setSignaledRole] = useState<PanelRole | null>(null);
  const [lines, setLines] = useState<SyncedTranscriptLine[]>([]);
  const [backchannelRoles, setBackchannelRoles] = useState<Record<string, boolean>>({});
  const [interruptNotice, setInterruptNotice] = useState<InterruptNotice | null>(null);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Agent line the next synced stream belongs to, and segments per streaming line.
  const openLineRef = useRef<{ id: string; fullText: string } | null>(null);
  const segmentsRef = useRef<Map<string, Map<string, string>>>(new Map());
  const pendingTookFloorRef = useRef<{ role: string; at: number } | null>(null);
  const timersRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());

  const later = useCallback((fn: () => void, ms: number) => {
    const timer = setTimeout(() => {
      timersRef.current.delete(timer);
      fn();
    }, ms);
    timersRef.current.add(timer);
  }, []);

  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach(clearTimeout);
      timers.clear();
    };
  }, []);

  const updateLine = useCallback(
    (id: string, patch: (line: SyncedTranscriptLine) => SyncedTranscriptLine) => {
      setLines((prev) => prev.map((line) => (line.id === id ? patch(line) : line)));
    },
    [],
  );

  /** Stops streaming a line; if no synced words ever arrived, shows its full text.
   * Late words of its last segment are still accepted for a while. */
  const settleLine = useCallback(
    (id: string, fullText: string) => {
      const streamed = segmentsRef.current.get(id);
      const hasWords = !!streamed && !!joinSegments(streamed);
      updateLine(id, (line) => ({
        ...line,
        streaming: false,
        text: hasWords ? line.text : line.text || fullText,
      }));
      if (!hasWords) segmentsRef.current.delete(id);
      else later(() => segmentsRef.current.delete(id), SEGMENT_RETAIN_MS);
    },
    [later, updateLine],
  );

  const writeSegment = useCallback(
    (lineId: string, segmentId: string, text: string) => {
      const segments = segmentsRef.current.get(lineId);
      if (!segments) return; // already settled
      segments.set(segmentId, text);
      const joined = joinSegments(segments);
      updateLine(lineId, (line) => ({ ...line, text: joined }));
    },
    [updateLine],
  );

  const agentIdentity = agentParticipant?.identity;

  useEffect(() => {
    if (!agentIdentity) return;
    const onStream = (reader: TextStreamReader, participant: { identity: string }) => {
      if (participant.identity !== agentIdentity) return;
      const target = openLineRef.current;
      if (!target) return;
      // A stream carries one segment: deltas in one stream, or full text per stream.
      const segmentId = reader.info.attributes?.['lk.segment_id'] ?? reader.info.id;
      void (async () => {
        let text = '';
        try {
          for await (const chunk of reader) {
            text += chunk;
            writeSegment(target.id, segmentId, text);
          }
        } catch (err) {
          console.warn('transcription stream failed', err);
        }
      })();
    };
    try {
      room.registerTextStreamHandler(TRANSCRIPTION_TOPIC, onStream);
    } catch (err) {
      console.warn('transcription handler already registered', err);
      return;
    }
    return () => room.unregisterTextStreamHandler(TRANSCRIPTION_TOPIC);
  }, [room, agentIdentity, writeSegment]);

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
    (role: string, speaker: string, text: string, opts: LineOptions = {}) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      const id = newId();
      const pending = pendingTookFloorRef.current;
      const tookFloor =
        !!pending && pending.role === role && Date.now() - pending.at < TOOK_FLOOR_WINDOW_MS;
      if (tookFloor) pendingTookFloorRef.current = null;

      if (opts.stream) {
        const previous = openLineRef.current;
        if (previous) settleLine(previous.id, previous.fullText);
        openLineRef.current = { id, fullText: trimmed };
        segmentsRef.current.set(id, new Map());
        later(() => {
          const segments = segmentsRef.current.get(id);
          if (segments && !joinSegments(segments)) settleLine(id, trimmed);
        }, STREAM_FALLBACK_MS);
      }
      const line: SyncedTranscriptLine = {
        id,
        role,
        speaker,
        text: opts.stream ? '' : trimmed,
        ts: Date.now(),
        streaming: !!opts.stream,
        tookFloor,
        backchannels: [],
      };
      setLines((prev) => [...prev, line]);
    },
    [later, settleLine],
  );

  const onInterrupt = useCallback(
    (ev: { role: string; name: string; by_role: string; by_name: string; heard_text?: string }) => {
      const notice = { id: Date.now(), name: ev.name, byName: ev.by_name };
      setInterruptNotice(notice);
      later(() => {
        setInterruptNotice((cur) => (cur?.id === notice.id ? null : cur));
      }, INTERRUPT_NOTICE_MS);
      pendingTookFloorRef.current = { role: ev.by_role, at: Date.now() };

      setLines((prev) => {
        const idx = prev.map((line) => line.role).lastIndexOf(ev.role);
        if (idx < 0) return prev;
        const next = [...prev];
        const line = next[idx];
        // The synced words stop where the audio was cut; fall back to the kept prefix.
        next[idx] = {
          ...line,
          interruptedBy: ev.by_name,
          text: line.text || (ev.heard_text ?? '').trim(),
        };
        return next;
      });
    },
    [later],
  );

  const onBackchannel = useCallback(
    (ev: { role: string; name: string; text: string; speaker_role: string; duration_s: number }) => {
      setBackchannelRoles((prev) => ({ ...prev, [ev.role]: true }));
      later(
        () => setBackchannelRoles((prev) => ({ ...prev, [ev.role]: false })),
        Math.max(0, ev.duration_s) * 1000 + BACKCHANNEL_TAIL_MS,
      );
      setLines((prev) => {
        const idx = prev.map((line) => line.role).lastIndexOf(ev.speaker_role);
        if (idx < 0) return prev;
        const next = [...prev];
        const line = next[idx];
        const marker: TranscriptBackchannel = {
          id: newId(),
          // Anchored where the synced words are right now, i.e. where it was heard.
          at: line.text.length,
          role: ev.role,
          name: ev.name,
          text: ev.text,
        };
        next[idx] = { ...line, backchannels: [...line.backchannels, marker] };
        return next;
      });
    },
    [later],
  );

  const lipSyncActive = useMemo(
    () => !!(signaledRole && signaledRole !== 'human' && lipSyncSpeaking),
    [signaledRole, lipSyncSpeaking],
  );

  return {
    activeRole,
    lipSyncActive,
    lines,
    backchannelRoles,
    interruptNotice,
    onRoleActive,
    onRoleIdle,
    pushLine,
    onInterrupt,
    onBackchannel,
  };
}
