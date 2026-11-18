'use client';

/**
 * Picks DyStream LiveKit, HTTP clip, or letter fallback avatar rendering for one AI panelist.
 */
import dynamic from 'next/dynamic';

import type { TrackReference } from '@livekit/components-core';
import type { PanelistDef } from '@/lib/talkshow/roles';
import { localIdleUrl, localPortraitUrl, AVATAR_ENGINE } from '@/lib/talkshow/avatarAssets';
import styles from '@/styles/TalkshowStage.module.css';

const AvatarPanelVideo = dynamic(
  () => import('@/lib/talkshow/AvatarPanelVideo').then((m) => ({ default: m.AvatarPanelVideo })),
  { ssr: false },
);

const AgentAvatarVideo = dynamic(
  () => import('@/lib/talkshow/AgentAvatarVideo').then((m) => ({ default: m.AgentAvatarVideo })),
  { ssr: false },
);

type Props = {
  panelist: PanelistDef;
  isSpeaking: boolean;
  isActive?: boolean;
  isWarming?: boolean;
  speechClipUrl?: string;
  agentVideoTrackRef?: TrackReference;
  /** The listener-face track, shown here while this panelist's backchannel plays. */
  backchannelTrackRef?: TrackReference;
  isBackchannel?: boolean;
  compact?: boolean;
};

function LetterAvatar({
  initial,
  color,
  compact,
  mode,
}: {
  initial: string;
  color: string;
  compact?: boolean;
  mode: string;
}) {
  return (
    <div
      className={styles.avatar}
      data-avatar-mode={mode}
      style={{
        background: color,
        ...(compact ? { width: 44, height: 44, fontSize: '1rem' } : {}),
      }}
    >
      {initial}
    </div>
  );
}

export function PanelAvatar({
  panelist,
  isSpeaking,
  isActive = false,
  isWarming = false,
  speechClipUrl,
  agentVideoTrackRef,
  backchannelTrackRef,
  isBackchannel = false,
  compact,
}: Props) {
  const avatar = panelist.avatar;
  const initial = panelist.name[0] ?? '?';
  const portraitUrl = localPortraitUrl(panelist.role) ?? avatar?.portrait;
  const idleVideoUrl = localIdleUrl(panelist.role) ?? avatar?.idle_video;

  if (process.env.NEXT_PUBLIC_DYSTREAM_ENABLED === '0') {
    return (
      <div className={styles.avatarName} data-avatar-mode="name" data-compact={compact}>
        {panelist.name}
      </div>
    );
  }

  // A backchannel while someone else holds the floor: this tile shows the listener track.
  const showBackchannel = Boolean(
    isBackchannel && !isActive && backchannelTrackRef?.publication?.track,
  );

  if (idleVideoUrl || agentVideoTrackRef) {
    return (
      <div
        className={styles.avatarSlot}
        data-avatar-mode={isActive ? 'dystream-livekit' : 'dystream-idle'}
        data-avatar-role={panelist.role}
        data-avatar-engine={AVATAR_ENGINE}
      >
        <AgentAvatarVideo
          trackRef={showBackchannel ? backchannelTrackRef : agentVideoTrackRef}
          idleVideoUrl={idleVideoUrl}
          isLive={isActive || showBackchannel}
          isWarming={isWarming}
          compact={compact}
          fallbackInitial={initial}
          fallbackColor={panelist.color}
        />
      </div>
    );
  }

  if (portraitUrl || speechClipUrl) {
    return (
      <div
        className={styles.avatarSlot}
        data-avatar-mode="dystream-idle"
        data-avatar-role={panelist.role}
      >
        <AvatarPanelVideo
          portraitUrl={portraitUrl}
          idleVideoUrl={idleVideoUrl}
          speechClipUrl={speechClipUrl}
          isSpeaking={isSpeaking}
          fallbackInitial={initial}
          fallbackColor={panelist.color}
          compact={compact}
        />
      </div>
    );
  }

  return (
    <LetterAvatar
      initial={initial}
      color={panelist.color}
      compact={compact}
      mode="letter"
    />
  );
}
