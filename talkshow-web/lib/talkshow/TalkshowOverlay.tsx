'use client';

import { useDataChannel, useLocalParticipant, useRemoteParticipants } from '@livekit/components-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { TranscriptPanel, type TranscriptLine } from '@/lib/talkshow/TranscriptPanel';
import { VirtualPanel, type HumanSeat } from '@/lib/talkshow/VirtualPanel';
import {
  UI_TOPIC,
  parseAgentMetadata,
  parseUiEvent,
  rosterFromParticipants,
  type PanelRole,
  type PanelistDef,
} from '@/lib/talkshow/roles';
import styles from '@/styles/TalkshowPanel.module.css';

function humanSeats(
  localName: string,
  localIdentity: string,
  remotes: Array<{ identity: string; name?: string; metadata?: string }>,
): HumanSeat[] {
  const seats: HumanSeat[] = [
    {
      key: localIdentity || 'local',
      name: localName || 'You',
      label: 'Human guest',
      isLocal: true,
    },
  ];
  for (const p of remotes) {
    const meta = parseAgentMetadata(p.metadata);
    if (meta?.talkshowAgent) continue;
    seats.push({
      key: p.identity,
      name: p.name || p.identity,
      label: 'Human guest',
    });
  }
  return seats;
}

/** Panel avatars + transcript; roster from agent panel_roster / participant metadata */
export function TalkshowOverlay() {
  const { localParticipant } = useLocalParticipant();
  const remotes = useRemoteParticipants();
  const [panelists, setPanelists] = useState<PanelistDef[]>([]);
  const [activeRole, setActiveRole] = useState<PanelRole | null>(null);
  const [lines, setLines] = useState<TranscriptLine[]>([]);

  const applyRoster = useCallback((members: PanelistDef[]) => {
    if (members.length) setPanelists(members);
  }, []);

  useEffect(() => {
    const fromMeta = rosterFromParticipants(remotes);
    if (fromMeta) applyRoster(fromMeta);
  }, [remotes, applyRoster]);

  const pushLine = useCallback((role: string, speaker: string, text: string) => {
    setLines((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${prev.length}`,
        role,
        speaker,
        text,
        ts: Date.now(),
      },
    ]);
  }, []);

  useDataChannel(UI_TOPIC, (msg) => {
    const ev = parseUiEvent(msg.payload);
    if (!ev) return;
    if (ev.type === 'panel_roster') {
      applyRoster(ev.members);
    } else if (ev.type === 'role_active') {
      setActiveRole(ev.role);
    } else if (ev.type === 'transcript' && ev.final) {
      pushLine(ev.role, ev.speaker, ev.text);
    }
  });

  const localName = localParticipant?.name || localParticipant?.identity || 'You';
  const localIdentity = localParticipant?.identity || 'local';
  const humans = useMemo(
    () =>
      humanSeats(
        localName,
        localIdentity,
        remotes.map((p) => ({
          identity: p.identity,
          name: p.name,
          metadata: p.metadata,
        })),
      ),
    [localName, localIdentity, remotes],
  );

  const waitingForAgent =
    panelists.length === 0 && !remotes.some((p) => parseAgentMetadata(p.metadata)?.talkshowAgent);

  return (
    <aside className={styles.overlay} aria-label="Talkshow panel">
      <VirtualPanel
        panelists={panelists}
        activeRole={activeRole}
        humans={humans}
        waitingForAgent={waitingForAgent}
      />
      <TranscriptPanel lines={lines} />
    </aside>
  );
}
