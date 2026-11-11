'use client';

/**
 * Human guest seat card with local camera preview and speaking state.
 */
import {
  VideoTrack,
  useIsSpeaking,
  useLocalParticipant,
  useRemoteParticipant,
} from '@livekit/components-react';
import { Track, type Participant } from 'livekit-client';
import { useMemo } from 'react';

import type { HumanSeat } from '@/lib/talkshow/VirtualPanel';
import type { PanelRole } from '@/lib/talkshow/roles';
import styles from '@/styles/TalkshowStage.module.css';

type Props = {
  human: HumanSeat;
  activeRole: PanelRole | null;
};

export function HumanSeatCard({ human, activeRole }: Props) {
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
    >
      {camTrackRef ? (
        <div className={styles.avatarSlot}>
          <VideoTrack trackRef={camTrackRef} className={styles.videoAvatar} />
        </div>
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

/** A human in the room seen from another seat (or the recording page): lit while
 * their microphone is speaking, or while the worker gives the floor to the human. */
export function RemoteHumanSeatCard({ human, activeRole }: Props) {
  const participant = useRemoteParticipant(human.key);
  return participant ? (
    <SpeakingRemoteCard human={human} activeRole={activeRole} participant={participant} />
  ) : (
    <RemoteCard human={human} speaking={activeRole === 'human'} />
  );
}

function SpeakingRemoteCard({
  human,
  activeRole,
  participant,
}: Props & { participant: Participant }) {
  const isSpeaking = useIsSpeaking(participant);
  return <RemoteCard human={human} speaking={isSpeaking || activeRole === 'human'} />;
}

function RemoteCard({ human, speaking }: { human: HumanSeat; speaking: boolean }) {
  return (
    <div className={styles.card} data-human data-speaking={speaking ? 'true' : 'false'}>
      <div className={styles.avatarHuman}>{human.name[0] ?? '?'}</div>
      <div className={styles.meta}>
        <strong>{human.name}</strong>
        <span>{human.label}</span>
      </div>
    </div>
  );
}
