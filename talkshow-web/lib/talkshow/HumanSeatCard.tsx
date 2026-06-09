'use client';

import { VideoTrack, useIsSpeaking, useLocalParticipant } from '@livekit/components-react';
import { Track } from 'livekit-client';
import { useMemo } from 'react';

import type { HumanSeat } from '@/lib/talkshow/VirtualPanel';
import type { PanelRole } from '@/lib/talkshow/roles';
import styles from '@/styles/TalkshowStage.module.css';

type Props = {
  human: HumanSeat;
  activeRole: PanelRole | null;
  handRaised?: boolean;
};

export function HumanSeatCard({ human, activeRole, handRaised }: Props) {
  const { localParticipant, cameraTrack } = useLocalParticipant();
  const isLocalSpeaking = useIsSpeaking(localParticipant);

  const speaking =
    human.isLocal && (isLocalSpeaking || activeRole === 'human');

  const camTrackRef = useMemo(() => {
    if (!human.isLocal || !cameraTrack || !localParticipant.isCameraEnabled) {
      return undefined;
    }
    return {
      participant: localParticipant,
      publication: cameraTrack,
      source: Track.Source.Camera,
    };
  }, [human.isLocal, cameraTrack, localParticipant, localParticipant.isCameraEnabled]);

  return (
    <div
      key={human.key}
      className={styles.card}
      data-human
      data-speaking={speaking ? 'true' : 'false'}
      data-hand={handRaised ? 'raised' : 'false'}
    >
      {handRaised && (
        <span className={styles.handBadge} title="Hand raised">
          ✋
        </span>
      )}
      {camTrackRef ? (
        <VideoTrack trackRef={camTrackRef} className={styles.videoAvatar} />
      ) : (
        <div className={styles.avatarHuman}>{human.isLocal ? 'You' : (human.name[0] ?? '?')}</div>
      )}
      <div className={styles.meta}>
        <strong>{human.name}</strong>
        <span>{human.label}</span>
      </div>
    </div>
  );
}
