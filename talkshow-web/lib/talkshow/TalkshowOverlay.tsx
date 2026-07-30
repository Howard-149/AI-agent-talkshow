'use client';

import { useDataChannel, useLocalParticipant, useRemoteParticipants } from '@livekit/components-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { TranscriptPanel } from '@/lib/talkshow/TranscriptPanel';
import {
  findAgentParticipant,
  usePlayoutSyncedUi,
} from '@/lib/talkshow/usePlayoutSyncedUi';
import { VirtualPanel, type HumanSeat } from '@/lib/talkshow/VirtualPanel';
import {
  UI_TOPIC,
  parseAgentMetadata,
  parseUiEvent,
  rosterFromParticipants,
  type PanelistDef,
} from '@/lib/talkshow/roles';
import { useAgentAvatarVideo } from '@/lib/talkshow/useAgentAvatarVideo';
import { useAvatarClips } from '@/lib/talkshow/useAvatarClips';
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
  const agentParticipant = useMemo(() => findAgentParticipant(remotes), [remotes]);
  const { activeRole, lipSyncActive, lines, onRoleActive, onRoleIdle, pushLine } =
    usePlayoutSyncedUi(agentParticipant);

  const applyRoster = useCallback((members: PanelistDef[]) => {
    if (members.length) setPanelists(members);
  }, []);

  useEffect(() => {
    const fromMeta = rosterFromParticipants(remotes);
    if (fromMeta) applyRoster(fromMeta);
  }, [remotes, applyRoster]);

  const { clipForRole, warmingRole, handleUiEvent } = useAvatarClips();
  const agentVideoTrackRef = useAgentAvatarVideo(agentParticipant);
  const handleUiEventRef = useRef(handleUiEvent);
  handleUiEventRef.current = handleUiEvent;
  const pushLineRef = useRef(pushLine);
  pushLineRef.current = pushLine;
  const onRoleActiveRef = useRef(onRoleActive);
  onRoleActiveRef.current = onRoleActive;
  const onRoleIdleRef = useRef(onRoleIdle);
  onRoleIdleRef.current = onRoleIdle;

  useDataChannel(UI_TOPIC, (msg) => {
    const ev = parseUiEvent(msg.payload);
    if (!ev) return;
    if (ev.type === 'panel_roster') {
      applyRoster(ev.members);
    } else if (ev.type === 'role_active') {
      onRoleActiveRef.current(ev.role);
    } else if (ev.type === 'role_idle') {
      onRoleIdleRef.current();
    } else if (ev.type === 'transcript' && ev.final) {
      pushLineRef.current(ev.role, ev.speaker, ev.text);
    }
    handleUiEventRef.current(ev);
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
        lipSyncActive={lipSyncActive}
        humans={humans}
        waitingForAgent={waitingForAgent}
        clipForRole={clipForRole}
        warmingRole={warmingRole}
        agentVideoTrackRef={agentVideoTrackRef}
        compactAvatars
      />
      <TranscriptPanel lines={lines} />
    </aside>
  );
}
