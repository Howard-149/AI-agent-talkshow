'use client';

import {
  RoomAudioRenderer,
  useConnectionState,
  useDataChannel,
  useLocalParticipant,
  useRemoteParticipants,
} from '@livekit/components-react';
import { ConnectionState } from 'livekit-client';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { HumanSeatCard } from '@/lib/talkshow/HumanSeatCard';
import { TalkshowMediaControls } from '@/lib/talkshow/TalkshowMediaControls';
import {
  UI_TOPIC,
  parseAgentMetadata,
  parseUiEvent,
  rosterFromParticipants,
  type PanelRole,
  type PanelistDef,
} from '@/lib/talkshow/roles';
import styles from '@/styles/TalkshowStage.module.css';

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
  const { localParticipant } = useLocalParticipant();
  const remotes = useRemoteParticipants();
  const connectionState = useConnectionState();

  const [panelists, setPanelists] = useState<PanelistDef[]>([]);
  const [activeRole, setActiveRole] = useState<PanelRole | null>(null);
  const [lines, setLines] = useState<
    Array<{ id: string; speaker: string; role: string; text: string; ts: number }>
  >([]);
  const [showTranscript, setShowTranscript] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

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

  const agentConnected = remotes.some((p) => parseAgentMetadata(p.metadata)?.talkshowAgent);
  const waitingForAgent = panelists.length === 0 && !agentConnected;
  const micEnabled = localParticipant?.isMicrophoneEnabled ?? false;
  const camEnabled = localParticipant?.isCameraEnabled ?? false;

  const statusText = useMemo(() => {
    if (connectionState === ConnectionState.Connecting) return 'Connecting…';
    if (connectionState !== ConnectionState.Connected) return 'Disconnected';
    if (!micEnabled && !camEnabled) return 'Use controls below — mic/camera off by default';
    if (!micEnabled) return 'Mic off — toggle Microphone when ready to speak';
    if (waitingForAgent) return 'Waiting for show agent…';
    return 'Live';
  }, [connectionState, micEnabled, camEnabled, waitingForAgent]);

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
              <HumanSeatCard key={h.key} human={h} activeRole={activeRole} />
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
              style={{ '--role-color': p.color } as React.CSSProperties}
            >
              <div className={styles.avatar} style={{ background: p.color }}>
                {p.name[0]}
              </div>
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
        <TalkshowMediaControls
          showSettings={showSettings}
          onToggleSettings={() => setShowSettings((v) => !v)}
          showTranscript={showTranscript}
          onToggleTranscript={() => setShowTranscript((v) => !v)}
          transcriptCount={lines.length}
        />
      </div>
    </div>
  );
}
