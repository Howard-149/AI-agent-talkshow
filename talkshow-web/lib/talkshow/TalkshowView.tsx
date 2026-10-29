'use client';

/**
 * Main talkshow stage: panel seats, media controls, locale filter, and floor / hand-raise UI.
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
import { TalkshowMediaControls } from '@/lib/talkshow/TalkshowMediaControls';
import { ReactivePanel } from './ReactivePanel';
import { readInterruption, type InterruptionNotice, type SpeakingTurn } from './interruption';
import { publishHandRaise, publishSetLocale, publishTextTurn } from '@/lib/talkshow/control';
import {
  pickLocalizedText,
  writeStoredViewerLocale,
  type TalkshowLocale,
} from '@/lib/talkshow/locale';
import { findAgentParticipant, usePlayoutSyncedUi } from '@/lib/talkshow/usePlayoutSyncedUi';
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

const textInputMode = process.env.NEXT_PUBLIC_TALKSHOW_TEXT_INPUT === '1';

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
  props: { viewerLocale?: TalkshowLocale; recordMode?: boolean; aiOnly?: boolean } = {},
) {
  const [viewerLocale, setViewerLocale] = useState<TalkshowLocale>(props.viewerLocale ?? 'en');
  const room = useRoomContext();
  const { localParticipant } = useLocalParticipant();
  const remotes = useRemoteParticipants();
  const connectionState = useConnectionState();

  const [panelists, setPanelists] = useState<PanelistDef[]>([]);
  const agentParticipant = useMemo(() => findAgentParticipant(remotes), [remotes]);
  const {
    activeRole,
    lipSyncActive,
    lines,
    onRoleActive,
    onRoleIdle,
    pushLine,
    clearLines,
    finishLine,
  } = usePlayoutSyncedUi(agentParticipant);
  const [showTranscript, setShowTranscript] = useState(!!props.recordMode || textInputMode);
  const [humanHasFloor, setHumanHasFloor] = useState(false);
  const [draft, setDraft] = useState('');
  const [sendError, setSendError] = useState('');
  const [reactiveReady, setReactiveReady] = useState(false);
  const [demoRunning, setDemoRunning] = useState(false);
  const [topic, setTopic] = useState(props.aiOnly ? 'Should students use AI for homework?' : '');
  const [interruptionNotice, setInterruptionNotice] = useState<InterruptionNotice | null>(null);
  const [interruptionMarks, setInterruptionMarks] = useState<
    Record<string, { kind: 'interrupted' | 'takeover'; name: string }>
  >({});
  const speakingTurnRef = useRef<SpeakingTurn | null>(null);
  const pendingTakeoverRef = useRef<InterruptionNotice | null>(null);
  useEffect(() => {
    if (!interruptionNotice) return;
    const timer = setTimeout(() => setInterruptionNotice(null), 8000);
    return () => clearTimeout(timer);
  }, [interruptionNotice]);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (showTranscript) transcriptEndRef.current?.scrollIntoView({ block: 'nearest' });
  }, [lines, showTranscript]);
  const [showSettings, setShowSettings] = useState(false);
  const [handRaised, setHandRaised] = useState<Record<string, boolean>>({});
  const [queueState, setQueueState] = useState<QueueEntry[]>([]);
  const [queuePhase, setQueuePhase] = useState('');
  const [floorPending, setFloorPending] = useState(false);
  const [showQueueDebug, setShowQueueDebug] = useState(false);
  const localHandUp = !!handRaised.human;

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
  const viewerLocaleRef = useRef(viewerLocale);
  viewerLocaleRef.current = viewerLocale;
  const openingHandRef = useRef(false);

  useDataChannel(UI_TOPIC, (msg) => {
    const ev = parseUiEvent(msg.payload);
    if (!ev) return;
    if (ev.type === 'reactive_ready') {
      setReactiveReady(ev.enabled);
    } else if (ev.type === 'reactive_reset') {
      setReactiveReady(true);
      clearLines();
      setHumanHasFloor(false);
      setSendError('');
      setDemoRunning(true);
      setTopic(ev.topic);
      setInterruptionNotice(null);
      setInterruptionMarks({});
      speakingTurnRef.current = null;
      pendingTakeoverRef.current = null;
    } else if (ev.type === 'reactive_decision' && props.aiOnly) {
      const notice = readInterruption(ev, speakingTurnRef.current);
      if (notice) {
        setInterruptionNotice(notice);
        pendingTakeoverRef.current = notice;
        setInterruptionMarks((prev) => ({
          ...prev,
          [`reactive-${notice.epoch}`]: { kind: 'interrupted', name: notice.byName },
        }));
      }
    } else if (ev.type === 'reactive_complete') {
      setDemoRunning(false);
    } else if (ev.type === 'reactive_state' || ev.type === 'reactive_generating') {
      // A viewer joining an active round may have missed the one-time ready
      // event. Actual reactive traffic is also an authoritative capability hint.
      setReactiveReady(true);
      if (ev.type === 'reactive_state' && ev.active) {
        speakingTurnRef.current = { role: ev.role, name: ev.name, epoch: ev.epoch };
        const takeover = pendingTakeoverRef.current;
        if (takeover && ev.epoch > takeover.epoch) {
          if (ev.role === takeover.byRole) {
            setInterruptionMarks((prev) => ({
              ...prev,
              [`reactive-${ev.epoch}`]: { kind: 'takeover', name: takeover.targetName },
            }));
          }
          pendingTakeoverRef.current = null;
        }
      }
      if (ev.type === 'reactive_state' && !ev.active) {
        if (speakingTurnRef.current?.epoch === ev.epoch) speakingTurnRef.current = null;
        finishLine(`reactive-${ev.epoch}`, !!ev.interrupted);
      }
    } else if (ev.type === 'transcript_partial') {
      if (viewerLocaleRef.current === 'en') {
        pushLineRef.current(ev.role, ev.speaker, ev.text, {
          id: ev.utterance_id,
          final: false,
        });
      }
    } else if (ev.type === 'reactive_error') {
      setSendError(ev.message);
      setDemoRunning(false);
    } else if (ev.type === 'panel_roster') {
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
      setHumanHasFloor(ev.role === 'human');
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
      // Like the video's ghost client, reserve the first human turn after
      // welcome so the panel does not debate an unspecified topic on joining.
      if (
        textInputMode &&
        !props.aiOnly &&
        ev.step === 'session_welcome' &&
        !openingHandRef.current
      ) {
        openingHandRef.current = true;
        void publishHandRaise(room, true).catch(() => {
          openingHandRef.current = false;
        });
      }
      const display = pickLocalizedText(ev.text, ev.texts, viewerLocaleRef.current);
      pushLineRef.current(ev.role, ev.speaker, display, {
        id: ev.utterance_id,
        final: true,
        interrupted: ev.interrupted,
      });
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

  const sendText = async () => {
    if (!humanHasFloor || !draft.trim() || !agentConnected) return;
    const text = draft.trim();
    setHumanHasFloor(false);
    setSendError('');
    try {
      await publishTextTurn(room, text);
      setDraft('');
    } catch {
      setHumanHasFloor(true);
      setSendError('Could not send. Please try again.');
    }
  };

  const statusText = useMemo(() => {
    if (connectionState === ConnectionState.Connecting) return 'Connecting…';
    if (connectionState !== ConnectionState.Connected) return 'Disconnected';
    if (props.aiOnly)
      return waitingForAgent ? 'Connecting studio…' : demoRunning ? 'Live' : 'Ready';
    if (textInputMode) {
      if (waitingForAgent) return 'Waiting for show agent…';
      return humanHasFloor ? 'Your turn — send your point' : 'Text input · AI voices live';
    }
    if (!micEnabled && !camEnabled) return 'Use controls below — mic/camera off by default';
    if (!micEnabled) return 'Mic off — toggle Microphone when ready to speak';
    if (floorPending) return 'Open floor — raise hand to speak';
    if (waitingForAgent) return 'Waiting for show agent…';
    return 'Live';
  }, [
    connectionState,
    micEnabled,
    camEnabled,
    waitingForAgent,
    floorPending,
    humanHasFloor,
    props.aiOnly,
    demoRunning,
  ]);

  return (
    <div className={`${styles.room} ${props.aiOnly ? styles.aiOnlyRoom : ''}`}>
      <RoomAudioRenderer />
      <header className={styles.header}>
        <h1 className={styles.title}>AI Talk Show</h1>
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
              disabled={textInputMode}
              onClick={() => changeLocale('zh')}
            >
              中文
            </button>
          </div>
          <span className={styles.status}>{statusText}</span>
        </div>
      </header>

      {topic && (
        <section className={styles.topicBanner} aria-label="Discussion topic">
          <span>Tonight’s topic</span>
          <h2>{topic}</h2>
        </section>
      )}

      <section
        className={`${styles.stage} ${reactiveReady ? styles.reactiveStage : ''}`}
        aria-label="Panel"
        data-talkshow-ready={panelists.length > 0 ? 'true' : 'false'}
      >
        <div className={styles.panelRow}>
          {props.aiOnly && (
            <div
              className={`${styles.card} ${styles.observerCard}`}
              data-observer
              data-speaking="false"
            >
              <span className={styles.observerBadge}>YOU</span>
              <div className={styles.avatarHuman} aria-label="Gefei Gu avatar">
                G
              </div>
              <div className={styles.meta}>
                <strong>Gefei Gu</strong>
                <span>Human · Observer</span>
              </div>
            </div>
          )}
          {!props.aiOnly &&
            humans.map((h) =>
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
              <span className={styles.aiSeatBadge} aria-label="AI character">
                AI
              </span>
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
        <div className={`${styles.body} ${reactiveReady ? styles.reactiveBody : ''}`}>
          <div className={styles.transcript}>
            <div className={styles.transcriptHeading}>
              <h3 className={styles.transcriptTitle}>Transcript</h3>
              {interruptionNotice && (
                <div className={styles.interruptionBanner} role="status" aria-live="polite">
                  <span>AI interruption</span>
                  <strong>
                    {interruptionNotice.byName} interrupts {interruptionNotice.targetName}
                  </strong>
                </div>
              )}
            </div>
            <div className={styles.transcriptScroll}>
              {lines.length === 0 && <p className={styles.muted}>Waiting for speech…</p>}
              {lines.map((line) => (
                <article
                  key={line.id}
                  className={styles.line}
                  data-role={line.role}
                  data-streaming={!line.final}
                  data-interrupted={!!line.interrupted}
                >
                  <header className={styles.lineHeader}>
                    <span className={styles.speaker}>{line.speaker}</span>
                    {line.role !== 'human' && <span className={styles.aiTranscriptBadge}>AI</span>}
                    {(line.interrupted || interruptionMarks[line.id]) && (
                      <span className={styles.interruptedLabel}>
                        {interruptionMarks[line.id]?.kind === 'interrupted'
                          ? `Interrupted by ${interruptionMarks[line.id].name}`
                          : interruptionMarks[line.id]?.kind === 'takeover'
                            ? 'Took the floor'
                            : 'Interrupted'}
                      </span>
                    )}
                    <time className={styles.time}>
                      {new Date(line.ts).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </time>
                  </header>
                  <p className={styles.lineText}>
                    {line.text}
                    {!line.final && (
                      <span className={styles.liveCursor} aria-label="Live caption" />
                    )}
                  </p>
                </article>
              ))}
              <div ref={transcriptEndRef} />
            </div>
          </div>
          {reactiveReady && (
            <ReactivePanel
              aiOnly={props.aiOnly}
              running={demoRunning}
              canStart={agentConnected && (props.aiOnly ? !demoRunning : humanHasFloor)}
              onStart={() => {
                setHumanHasFloor(false);
                setDemoRunning(true);
              }}
            />
          )}
        </div>
      )}

      {!props.recordMode && !props.aiOnly && (
        <div className={styles.controlDock}>
          {textInputMode && !isGhostViewer(localIdentity) && (
            <form
              className={styles.textInput}
              onSubmit={(event) => {
                event.preventDefault();
                void sendText();
              }}
            >
              <label htmlFor="talkshow-point">
                {humanHasFloor ? 'Your turn' : 'Raise your hand to join the discussion'}
              </label>
              <div className={styles.textInputRow}>
                <input
                  id="talkshow-point"
                  aria-label="Your point"
                  value={draft}
                  maxLength={2000}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="Type your point…"
                />
                <button
                  type="submit"
                  className="lk-button"
                  disabled={!humanHasFloor || !draft.trim() || !agentConnected}
                >
                  Send
                </button>
              </div>
              {sendError && <p role="alert">{sendError}</p>}
            </form>
          )}
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
            viewerLocale={viewerLocale}
            onLocaleChange={changeLocale}
          />
        </div>
      )}
    </div>
  );
}
