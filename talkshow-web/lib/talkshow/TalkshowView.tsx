'use client';

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
import { TalkshowMediaControls } from '@/lib/talkshow/TalkshowMediaControls';
import { publishHandRaise } from '@/lib/talkshow/control';
import {
  findAgentParticipant,
  usePlayoutSyncedUi,
} from '@/lib/talkshow/usePlayoutSyncedUi';
import {
  UI_TOPIC,
  parseAgentMetadata,
  parseUiEvent,
  rosterFromParticipants,
  type PanelistDef,
} from '@/lib/talkshow/roles';
import { useAgentAvatarVideo } from '@/lib/talkshow/useAgentAvatarVideo';
import { useAvatarClips } from '@/lib/talkshow/useAvatarClips';
import styles from '@/styles/TalkshowStage.module.css';

export type QueueEntry = {
  role: string;
  name: string;
  reason?: string;
  topic?: string;
};

function handStateFromQueue(queue: QueueEntry[]): Record<string, boolean> {
  const next: Record<string, boolean> = {};
  for (const item of queue) {
    next[item.role] = true;
  }
  return next;
}

function mergeQueueEntry(
  queue: QueueEntry[],
  entry: { role: string; name: string; reason?: string; topic?: string },
  raised: boolean,
): QueueEntry[] {
  if (!raised) {
    return queue.filter((item) => item.role !== entry.role);
  }
  const idx = queue.findIndex((item) => item.role === entry.role);
  const next: QueueEntry = {
    role: entry.role,
    name: entry.name,
    reason: entry.reason,
    topic: entry.topic,
  };
  if (idx >= 0) {
    const copy = [...queue];
    copy[idx] = { ...copy[idx], ...next };
    return copy;
  }
  return [...queue, next];
}

function humanSeats(
  localName: string,
  localIdentity: string,
  remotes: Array<{ identity: string; name?: string; metadata?: string }>,
) {
  const seats = [
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
      isLocal: false,
    });
  }
  return seats;
}

/** Virtual panel stage + LiveKit media controls (mic/camera/settings from Meet). */
export function TalkshowView() {
  const room = useRoomContext();
  const { localParticipant } = useLocalParticipant();
  const remotes = useRemoteParticipants();
  const connectionState = useConnectionState();

  const [panelists, setPanelists] = useState<PanelistDef[]>([]);
  const agentParticipant = useMemo(() => findAgentParticipant(remotes), [remotes]);
  const { activeRole, lipSyncActive, lines, onRoleActive, onRoleIdle, pushLine } =
    usePlayoutSyncedUi(agentParticipant);
  const [showTranscript, setShowTranscript] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [handRaised, setHandRaised] = useState<Record<string, boolean>>({});
  const [queueState, setQueueState] = useState<QueueEntry[]>([]);
  const [queuePhase, setQueuePhase] = useState('');
  const [floorPending, setFloorPending] = useState(false);
  const [showQueueDebug, setShowQueueDebug] = useState(false);
  const localHandUp = !!handRaised.human;

  const applyRoster = useCallback((members: PanelistDef[]) => {
    if (members.length) setPanelists(members);
  }, []);

  useEffect(() => {
    const fromMeta = rosterFromParticipants(remotes);
    if (fromMeta) applyRoster(fromMeta);
  }, [remotes, applyRoster]);

  const { clipForRole, warmingRole, handleUiEvent } = useAvatarClips();
  const agentVideoTrackRef = useAgentAvatarVideo(agentParticipant);

  const toggleHandRaise = useCallback(async () => {
    if (connectionState !== ConnectionState.Connected) return;
    const next = !localHandUp;
    try {
      await publishHandRaise(room, next);
      setHandRaised((prev) => ({ ...prev, human: next }));
    } catch (err) {
      console.error('hand_raise publish failed', err);
    }
  }, [connectionState, localHandUp, room]);

  const pushLineRef = useRef(pushLine);
  pushLineRef.current = pushLine;
  const onRoleActiveRef = useRef(onRoleActive);
  onRoleActiveRef.current = onRoleActive;
  const onRoleIdleRef = useRef(onRoleIdle);
  onRoleIdleRef.current = onRoleIdle;
  const handleUiEventRef = useRef(handleUiEvent);
  handleUiEventRef.current = handleUiEvent;

  useDataChannel(UI_TOPIC, (msg) => {
    const ev = parseUiEvent(msg.payload);
    if (!ev) return;
    if (ev.type === 'panel_roster') {
      applyRoster(ev.members);
    } else if (ev.type === 'role_active') {
      onRoleActiveRef.current(ev.role);
    } else if (ev.type === 'role_idle') {
      onRoleIdleRef.current();
    } else if (ev.type === 'hand_raise') {
      setHandRaised((prev) => ({ ...prev, [ev.role]: ev.raised }));
      setQueueState((prev) =>
        mergeQueueEntry(
          prev,
          { role: ev.role, name: ev.name, reason: ev.reason, topic: ev.topic },
          ev.raised,
        ),
      );
    } else if (ev.type === 'queue_state') {
      setQueueState(ev.queue);
      setQueuePhase(ev.phase ?? '');
      setHandRaised(handStateFromQueue(ev.queue));
    } else if (ev.type === 'floor_grant') {
      setHandRaised((prev) => {
        const next = { ...prev };
        delete next[ev.role];
        return next;
      });
      setQueueState((prev) => prev.filter((item) => item.role !== ev.role));
      setFloorPending(false);
    } else if (ev.type === 'floor_pending') {
      setFloorPending(!!ev.active);
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

  const agentConnected = remotes.some((p) => parseAgentMetadata(p.metadata)?.talkshowAgent);
  const waitingForAgent = panelists.length === 0 && !agentConnected;
  const micEnabled = localParticipant?.isMicrophoneEnabled ?? false;
  const camEnabled = localParticipant?.isCameraEnabled ?? false;

  const statusText = useMemo(() => {
    if (connectionState === ConnectionState.Connecting) return 'Connecting…';
    if (connectionState !== ConnectionState.Connected) return 'Disconnected';
    if (!micEnabled && !camEnabled) return 'Use controls below — mic/camera off by default';
    if (!micEnabled) return 'Mic off — toggle Microphone when ready to speak';
    if (floorPending) return 'Open floor — raise hand to speak';
    if (waitingForAgent) return 'Waiting for show agent…';
    return 'Live';
  }, [connectionState, micEnabled, camEnabled, waitingForAgent, floorPending]);

  return (
    <div className={styles.room}>
      <RoomAudioRenderer />
      <header className={styles.header}>
        <h1 className={styles.title}>AI Agent Talkshow</h1>
        <div className={styles.headerActions}>
          <span className={styles.status}>{statusText}</span>
        </div>
      </header>

      <section className={styles.stage} aria-label="Panel">
        <div className={styles.panelRow}>
          {humans.map((h) =>
            h.isLocal ? (
              <HumanSeatCard
                key={h.key}
                human={h}
                activeRole={activeRole}
                handRaised={localHandUp}
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
              data-hand={handRaised[p.role] ? 'raised' : 'false'}
              style={{ '--role-color': p.color } as React.CSSProperties}
            >
              {handRaised[p.role] && (
                <span className={styles.handBadge} title="Hand raised">
                  ✋
                </span>
              )}
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
            </div>
          ))}
        </div>
        {waitingForAgent && <p className={styles.waiting}>Show agent connecting…</p>}
      </section>

      {showTranscript && (
        <div className={styles.body}>
          <div className={styles.transcript}>
            <h3 className={styles.transcriptTitle}>Transcript</h3>
            <div className={styles.transcriptScroll}>
              {lines.length === 0 && <p className={styles.muted}>Waiting for speech…</p>}
              {lines.map((line) => (
                <article key={line.id} className={styles.line} data-role={line.role}>
                  <header className={styles.lineHeader}>
                    <span className={styles.speaker}>{line.speaker}</span>
                    <time className={styles.time}>
                      {new Date(line.ts).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </time>
                  </header>
                  <p className={styles.lineText}>{line.text}</p>
                </article>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className={styles.controlDock}>
        {showQueueDebug && (
          <div
            id="talkshow-hand-queue"
            className={styles.queuePanel}
            role="region"
            aria-label="Hand raise queue"
          >
            <h3 className={styles.queueTitle}>
              Hand raise queue
              {queuePhase ? <span className={styles.queuePhase}> · {queuePhase}</span> : null}
            </h3>
            {queueState.length === 0 ? (
              <p className={styles.muted}>Queue empty — waiting for hand raises</p>
            ) : (
              <ol className={styles.queueList}>
                {queueState.map((item, idx) => (
                  <li key={`${item.role}-${idx}`} className={styles.queueItem}>
                    <span className={styles.queuePos}>{idx + 1}</span>
                    <strong>{item.name}</strong>
                    <span className={styles.queueRole}>{item.role}</span>
                    {item.topic ? <span className={styles.queueMeta}>{item.topic}</span> : null}
                    {!item.topic && item.reason ? (
                      <span className={styles.queueMeta}>{item.reason}</span>
                    ) : null}
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}

        <button
          type="button"
          className={`${styles.queueBar} ${showQueueDebug ? styles.queueBarOpen : ''}`}
          onClick={() => setShowQueueDebug((v) => !v)}
          aria-expanded={showQueueDebug}
          aria-controls="talkshow-hand-queue"
        >
          <span className={styles.queueBarLabel}>
            ✋ Hand raise queue
            {queueState.length > 0 ? (
              <span className={styles.queueBarCount}>{queueState.length} waiting</span>
            ) : (
              <span className={styles.queueBarEmpty}>empty</span>
            )}
          </span>
          <span className={styles.queueBarChevron}>{showQueueDebug ? '▾' : '▸'}</span>
        </button>

        <TalkshowMediaControls
          showSettings={showSettings}
          onToggleSettings={() => setShowSettings((v) => !v)}
          showTranscript={showTranscript}
          onToggleTranscript={() => setShowTranscript((v) => !v)}
          transcriptCount={lines.length}
          handRaised={localHandUp}
          onToggleHandRaise={() => void toggleHandRaise()}
        />
      </div>
    </div>
  );
}
