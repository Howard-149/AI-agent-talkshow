'use client';

import { useEffect, useRef } from 'react';
import { VideoTrack } from '@livekit/components-react';
import type { TrackReference } from '@livekit/components-core';

import styles from '@/styles/TalkshowStage.module.css';

type Props = {
  trackRef?: TrackReference;
  idleVideoUrl?: string;
  /** Reveal synth + hide idle (playout started). */
  isLive?: boolean;
  /** Mount VideoTrack early (avatar_clip) so preroll→play does not drop frame 0. */
  isWarming?: boolean;
  compact?: boolean;
  fallbackInitial: string;
  fallbackColor: string;
};

/**
 * Idle loop while waiting; attach LiveKit during warm; on isLive cut to synth
 * immediately (no decoder-ready gate — that skipped the first spoken frames).
 */
export function AgentAvatarVideo({
  trackRef,
  idleVideoUrl,
  isLive = false,
  isWarming = false,
  compact = false,
  fallbackInitial,
  fallbackColor,
}: Props) {
  const idleRef = useRef<HTMLVideoElement>(null);

  const attachTrack = Boolean(
    trackRef?.publication?.track && (isLive || isWarming),
  );
  const showSynth = Boolean(isLive && trackRef?.publication?.track);

  useEffect(() => {
    const idle = idleRef.current;
    if (!idle || !idleVideoUrl) return;
    idle.play().catch(() => {});
  }, [idleVideoUrl]);

  if (!idleVideoUrl && !attachTrack) {
    return (
      <div
        className={styles.avatar}
        style={{
          background: fallbackColor,
          ...(compact ? { fontSize: '1rem' } : {}),
        }}
      >
        {fallbackInitial}
      </div>
    );
  }

  return (
    <div
      className={styles.avatarVideoWrap}
      data-avatar-live={showSynth ? 'true' : 'false'}
      data-avatar-warm={attachTrack && !showSynth ? 'true' : 'false'}
    >
      {idleVideoUrl ? (
        <video
          ref={idleRef}
          className={styles.avatarVideo}
          data-avatar-idle="1"
          src={idleVideoUrl}
          muted
          loop
          playsInline
          autoPlay
          style={{ opacity: showSynth ? 0 : 1 }}
        />
      ) : null}
      {attachTrack ? (
        <div
          className={styles.avatarSynthLayer}
          style={{ opacity: showSynth ? 1 : 0 }}
        >
          <VideoTrack trackRef={trackRef!} className={styles.avatarVideo} />
        </div>
      ) : null}
    </div>
  );
}
