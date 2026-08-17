'use client';

/**
 * Portrait, idle loop, and HTTP speech-clip video for panel seats without a LiveKit avatar track.
 */
import { useEffect, useRef, useState } from 'react';

import styles from '@/styles/TalkshowStage.module.css';

type Props = {
  portraitUrl?: string;
  idleVideoUrl?: string;
  speechClipUrl?: string;
  isSpeaking: boolean;
  fallbackInitial: string;
  fallbackColor: string;
  compact?: boolean;
};

export function AvatarPanelVideo({
  portraitUrl,
  idleVideoUrl,
  speechClipUrl,
  isSpeaking,
  fallbackInitial,
  fallbackColor,
  compact = false,
}: Props) {
  const idleRef = useRef<HTMLVideoElement>(null);
  const speechRef = useRef<HTMLVideoElement>(null);
  const [videoFailed, setVideoFailed] = useState(false);

  useEffect(() => {
    const idle = idleRef.current;
    if (!idle || !idleVideoUrl) return;
    idle.play().catch(() => {});
  }, [idleVideoUrl]);

  useEffect(() => {
    const speech = speechRef.current;
    if (!speech || !speechClipUrl || !isSpeaking) return;
    speech.currentTime = 0;
    speech.play().catch(() => {});
  }, [speechClipUrl, isSpeaking]);

  if (videoFailed || (!idleVideoUrl && !portraitUrl)) {
    return (
      <div
        className={styles.avatar}
        style={{
          background: fallbackColor,
          ...(compact ? { width: 44, height: 44, fontSize: '1rem' } : {}),
        }}
      >
        {fallbackInitial}
      </div>
    );
  }

  const showSpeech = isSpeaking && !!speechClipUrl;
  const wrapStyle = compact ? { width: 44, height: 44 } : undefined;

  return (
    <div className={styles.avatarVideoWrap} style={wrapStyle}>
      {portraitUrl && !idleVideoUrl && (
        <img
          className={styles.avatarPortrait}
          src={portraitUrl}
          alt=""
          onError={() => setVideoFailed(true)}
        />
      )}
      {idleVideoUrl && (
        <video
          ref={idleRef}
          className={styles.avatarVideo}
          src={idleVideoUrl}
          muted
          loop
          playsInline
          autoPlay
          style={{ opacity: showSpeech ? 0 : 1 }}
          onError={() => setVideoFailed(true)}
        />
      )}
      {speechClipUrl && (
        <video
          ref={speechRef}
          className={styles.avatarVideo}
          src={speechClipUrl}
          muted
          playsInline
          style={{ opacity: showSpeech ? 1 : 0 }}
          onError={() => setVideoFailed(true)}
        />
      )}
    </div>
  );
}
