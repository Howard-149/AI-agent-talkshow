'use client';

/**
 * Compact overlay stage: wires roster, avatars, and playout-synced transcript for the talkshow UI.
 */
import { useDataChannel, useLocalParticipant, useRemoteParticipants } from '@livekit/components-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { TranscriptPanel } from '@/lib/talkshow/TranscriptPanel';
import {
  pickLocalizedText,
  readStoredViewerLocale,
} from '@/lib/talkshow/locale';
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
  const {
    activeRole,
    lipSyncActive,
    lines,
    backchannelRoles,
    interruptNotice,
    onRoleActive,
    onRoleIdle,
    pushLine,
    onLineEnd,
    onInterrupt,
    onBackchannel,
  } = usePlayoutSyncedUi(agentParticipant);

  const applyRoster = useCallback((members: PanelistDef[]) => {
    if (members.length) setPanelists(members);
  }, []);

  useEffect(() => {
    const fromMeta = rosterFromParticipants(remotes);
    if (fromMeta) applyRoster(fromMeta);
  }, [remotes, applyRoster]);

  const { clipForRole, warmingRole, handleUiEvent } = useAvatarClips();
  const viewerLocale = readStoredViewerLocale();
  const agentVideoTrackRef = useAgentAvatarVideo(agentParticipant, viewerLocale);
  const handleUiEventRef = useRef(handleUiEvent);
  handleUiEventRef.current = handleUiEvent;
  const pushLineRef = useRef(pushLine);
  pushLineRef.current = pushLine;
  const onLineEndRef = useRef(onLineEnd);
  onLineEndRef.current = onLineEnd;
  const onInterruptRef = useRef(onInterrupt);
  onInterruptRef.current = onInterrupt;
  const onBackchannelRef = useRef(onBackchannel);
  onBackchannelRef.current = onBackchannel;
  const onRoleActiveRef = useRef(onRoleActive);
  onRoleActiveRef.current = onRoleActive;
  const onRoleIdleRef = useRef(onRoleIdle);
  onRoleIdleRef.current = onRoleIdle;
  const viewerLocaleRef = useRef(viewerLocale);
  viewerLocaleRef.current = viewerLocale;

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
      const display = pickLocalizedText(ev.text, ev.texts, viewerLocaleRef.current);
      pushLineRef.current(ev.role, ev.speaker, display, {
        stream: !!ev.stream && display === ev.text,
        step: ev.step,
      });
    } else if (ev.type === 'line_end') {
      onLineEndRef.current(ev);
    } else if (ev.type === 'interrupt') {
      onInterruptRef.current(ev);
    } else if (ev.type === 'backchannel') {
      onBackchannelRef.current(ev);
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
        backchannelRoles={backchannelRoles}
        humans={humans}
        waitingForAgent={waitingForAgent}
        clipForRole={clipForRole}
        warmingRole={warmingRole}
        agentVideoTrackRef={agentVideoTrackRef}
        compactAvatars
      />
      <TranscriptPanel lines={lines} interruptNotice={interruptNotice} />
    </aside>
  );
}
