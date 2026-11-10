'use client';

/**
 * Main talkshow stage: panel seats, media controls, locale filter, and transcript.
 */
import {
  RoomAudioRenderer,
  useConnectionState,
  useDataChannel,
  useLocalParticipant,
  useRemoteParticipants,
  useRoomContext,
} from '@livekit/components-react';
import { ConnectionState } from 'livekit-client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { HumanSeatCard } from '@/lib/talkshow/HumanSeatCard';
import { PanelAvatar } from '@/lib/talkshow/PanelAvatar';
import { PadMeter, type PadState } from '@/lib/talkshow/PadMeter';
import { TalkshowMediaControls } from '@/lib/talkshow/TalkshowMediaControls';
import { TranscriptPanel } from '@/lib/talkshow/TranscriptPanel';
import { publishSetLocale } from '@/lib/talkshow/control';
import {
  pickLocalizedText,
  writeStoredViewerLocale,
  type TalkshowLocale,
} from '@/lib/talkshow/locale';
import {
  findAgentParticipant,
  usePlayoutSyncedUi,
} from '@/lib/talkshow/usePlayoutSyncedUi';
import {
  UI_TOPIC,
  isGhostViewer,
  parseAgentMetadata,
  parseUiEvent,
  rosterFromParticipants,
  type PanelistDef,
} from '@/lib/talkshow/roles';
import { useAgentAvatarVideo } from '@/lib/talkshow/useAgentAvatarVideo';
import { useAvatarClips } from '@/lib/talkshow/useAvatarClips';
import { useLocaleMediaFilter } from '@/lib/talkshow/useLocaleMediaFilter';
import styles from '@/styles/TalkshowStage.module.css';

function humanSeats(
  localName: string,
  localIdentity: string,
  remotes: Array<{ identity: string; name?: string; metadata?: string }>,
) {
  const seats: Array<{
    key: string;
    name: string;
    label: string;
    isLocal: boolean;
  }> = [];
  if (!isGhostViewer(localIdentity)) {
    seats.push({
      key: localIdentity || 'local',
      name: localName || 'You',
      label: 'Human guest',
      isLocal: true,
    });
  }
  for (const p of remotes) {
    const meta = parseAgentMetadata(p.metadata);
    if (meta?.talkshowAgent) continue;
    if (isGhostViewer(p.identity, p.metadata)) continue;
    seats.push({
      key: p.identity,
      name: p.name || p.identity,
      label: 'Human guest',
      isLocal: false,
    });
  }
  return seats;
}

/** Virtual panel stage + LiveKit media controls (mic/camera/settings from Meet). */
export function TalkshowView(
  props: { viewerLocale?: TalkshowLocale; recordMode?: boolean } = {},
) {
  const [viewerLocale, setViewerLocale] = useState<TalkshowLocale>(
    props.viewerLocale ?? 'en',
  );
  const room = useRoomContext();
  const { localParticipant } = useLocalParticipant();
  const remotes = useRemoteParticipants();
  const connectionState = useConnectionState();

  const [panelists, setPanelists] = useState<PanelistDef[]>([]);
  const agentParticipant = useMemo(() => findAgentParticipant(remotes), [remotes]);
  const topic = parseAgentMetadata(agentParticipant?.metadata)?.topic?.trim() ?? '';
  const {
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
  } = usePlayoutSyncedUi(agentParticipant);
  const [showTranscript, setShowTranscript] = useState(!!props.recordMode);
  const [showSettings, setShowSettings] = useState(false);
  const [floorPending, setFloorPending] = useState(false);
  const [padByRole, setPadByRole] = useState<Record<string, PadState>>({});

  useEffect(() => {
    if (props.viewerLocale) setViewerLocale(props.viewerLocale);
  }, [props.viewerLocale]);

  useLocaleMediaFilter(viewerLocale);

  useEffect(() => {
    if (connectionState !== ConnectionState.Connected) return;
    void publishSetLocale(room, viewerLocale).catch((err) => {
      console.error('set_locale publish failed', err);
    });
  }, [connectionState, room, viewerLocale]);

  const changeLocale = useCallback(
    (next: TalkshowLocale) => {
      if (next === viewerLocale) return;
      setViewerLocale(next);
      writeStoredViewerLocale(next);
      if (connectionState !== ConnectionState.Connected) return;
      void publishSetLocale(room, next).catch((err) => {
        console.error('locale change failed', err);
      });
    },
    [viewerLocale, connectionState, room],
  );

  const applyRoster = useCallback((members: PanelistDef[]) => {
    if (members.length) setPanelists(members);
  }, []);

  useEffect(() => {
    const fromMeta = rosterFromParticipants(remotes);
    if (fromMeta) applyRoster(fromMeta);
  }, [remotes, applyRoster]);

  const { clipForRole, warmingRole, handleUiEvent } = useAvatarClips();
  const agentVideoTrackRef = useAgentAvatarVideo(agentParticipant, viewerLocale);

  const pushLineRef = useRef(pushLine);
  pushLineRef.current = pushLine;
  const onInterruptRef = useRef(onInterrupt);
  onInterruptRef.current = onInterrupt;
  const onBackchannelRef = useRef(onBackchannel);
  onBackchannelRef.current = onBackchannel;
  const onRoleActiveRef = useRef(onRoleActive);
  onRoleActiveRef.current = onRoleActive;
  const onRoleIdleRef = useRef(onRoleIdle);
  onRoleIdleRef.current = onRoleIdle;
  const handleUiEventRef = useRef(handleUiEvent);
  handleUiEventRef.current = handleUiEvent;
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
    } else if (ev.type === 'floor_grant') {
      setFloorPending(false);
    } else if (ev.type === 'pad_state') {
      setPadByRole((prev) => ({
        ...prev,
        [ev.role]: {
          pad: ev.pad,
          emotion: ev.emotion,
          // Expression-only updates carry no feeling; keep the last appraisal's phrase.
          feeling: ev.feeling ?? prev[ev.role]?.feeling,
        },
      }));
    } else if (ev.type === 'floor_pending') {
      setFloorPending(!!ev.active);
    } else if (ev.type === 'transcript' && ev.final) {
      const display = pickLocalizedText(ev.text, ev.texts, viewerLocaleRef.current);
      // Synced words are in the delivered language; another locale shows the whole line.
      pushLineRef.current(ev.role, ev.speaker, display, {
        stream: !!ev.stream && display === ev.text,
      });
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

  const agentConnected = remotes.some((p) => parseAgentMetadata(p.metadata)?.talkshowAgent);
  const waitingForAgent = panelists.length === 0 && !agentConnected;
  const micEnabled = localParticipant?.isMicrophoneEnabled ?? false;
  const camEnabled = localParticipant?.isCameraEnabled ?? false;

  const statusText = useMemo(() => {
    if (connectionState === ConnectionState.Connecting) return 'Connecting…';
    if (connectionState !== ConnectionState.Connected) return 'Disconnected';
    if (!micEnabled && !camEnabled) return 'Use controls below — mic/camera off by default';
    if (!micEnabled) return 'Mic off — toggle Microphone when ready to speak';
    if (floorPending) return 'Open floor — unmute and speak to join';
    if (waitingForAgent) return 'Waiting for show agent…';
    return 'Live';
  }, [connectionState, micEnabled, camEnabled, waitingForAgent, floorPending]);

  return (
    <div className={styles.room}>
      <RoomAudioRenderer />
      {topic && (
        <div className={styles.topicBanner} role="note" aria-label="Discussion topic">
          <span className={styles.topicLabel}>Topic</span>
          <span className={styles.topicText}>{topic}</span>
        </div>
      )}
      <header className={styles.header}>
        <h1 className={styles.title}>AI Agent Talkshow</h1>
        <div className={styles.headerActions}>
          <div className={styles.localeToggle} role="group" aria-label="Language">
            <button
              type="button"
              className={`${styles.localeBtn} ${viewerLocale === 'en' ? styles.localeBtnOn : ''}`}
              aria-pressed={viewerLocale === 'en'}
              onClick={() => changeLocale('en')}
            >
              EN
            </button>
            <button
              type="button"
              className={`${styles.localeBtn} ${viewerLocale === 'zh' ? styles.localeBtnOn : ''}`}
              aria-pressed={viewerLocale === 'zh'}
              onClick={() => changeLocale('zh')}
            >
              中文
            </button>
          </div>
          <span className={styles.status}>{statusText}</span>
        </div>
      </header>

      <section
        className={styles.stage}
        aria-label="Panel"
        data-talkshow-ready={panelists.length > 0 ? 'true' : 'false'}
      >
        <div className={styles.panelRow}>
          {humans.map((h) =>
            h.isLocal ? (
              <HumanSeatCard
                key={h.key}
                human={h}
                activeRole={activeRole}
              />
            ) : (
              <div
                key={h.key}
                className={styles.card}
                data-human
                data-speaking={activeRole === 'human' ? 'true' : 'false'}
              >
                <div className={styles.avatarHuman}>{h.name[0] ?? '?'}</div>
                <div className={styles.meta}>
                  <strong>{h.name}</strong>
                  <span>{h.label}</span>
                </div>
              </div>
            ),
          )}
          {panelists.map((p) => (
            <div
              key={p.role}
              className={styles.card}
              data-speaking={activeRole === p.role ? 'true' : 'false'}
              data-backchannel={backchannelRoles[p.role] ? 'true' : 'false'}
              style={{ '--role-color': p.color } as React.CSSProperties}
            >
              <PanelAvatar
                panelist={p}
                isSpeaking={activeRole === p.role && lipSyncActive}
                isActive={activeRole === p.role}
                isWarming={warmingRole === p.role}
                speechClipUrl={clipForRole(p.role)}
                agentVideoTrackRef={agentVideoTrackRef}
              />
              <div className={styles.meta}>
                <strong>{p.name}</strong>
                <span>{p.label}</span>
              </div>
              <PadMeter state={padByRole[p.role]} />
            </div>
          ))}
        </div>
        {waitingForAgent && <p className={styles.waiting}>Show agent connecting…</p>}
      </section>

      {showTranscript && (
        <div className={styles.body}>
          <TranscriptPanel lines={lines} interruptNotice={interruptNotice} />
        </div>
      )}

      {!props.recordMode && (
        <div className={styles.controlDock}>
        <TalkshowMediaControls
          showSettings={showSettings}
          onToggleSettings={() => setShowSettings((v) => !v)}
          showTranscript={showTranscript}
          onToggleTranscript={() => setShowTranscript((v) => !v)}
          transcriptCount={lines.length}
          viewerLocale={viewerLocale}
          onLocaleChange={changeLocale}
        />
        </div>
      )}
    </div>
  );
}
