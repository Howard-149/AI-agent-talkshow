'use client';

import { useTracks } from '@livekit/components-react';
import type { TrackReference } from '@livekit/components-core';
import { Track } from 'livekit-client';
import type { Participant } from 'livekit-client';
import { useMemo } from 'react';

const AVATAR_TRACK_NAME = 'talkshow-avatar';

/** Agent-published DyStream video track (muted; audio stays on agent audio track). */
export function useAgentAvatarVideo(
  agentParticipant: Participant | undefined,
): TrackReference | undefined {
  const cameraTracks = useTracks([Track.Source.Camera], {
    onlySubscribed: true,
  });

  return useMemo(() => {
    if (!agentParticipant) return undefined;

    const named = cameraTracks.find(
      (ref) =>
        ref.participant.sid === agentParticipant.sid &&
        ref.publication?.trackName === AVATAR_TRACK_NAME &&
        ref.publication?.track,
    );
    if (named) return named;

    // Fallback: agent's only camera track
    const anyCam = cameraTracks.find(
      (ref) =>
        ref.participant.sid === agentParticipant.sid && ref.publication?.track,
    );
    return anyCam;
  }, [cameraTracks, agentParticipant]);
}
