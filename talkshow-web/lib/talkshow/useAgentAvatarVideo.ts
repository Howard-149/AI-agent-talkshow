'use client';

/**
 * Resolves the agent-published DyStream avatar video track for the viewer's preferred locale.
 */
import { useTracks } from '@livekit/components-react';
import type { TrackReference } from '@livekit/components-core';
import { Track } from 'livekit-client';
import type { Participant } from 'livekit-client';
import { useMemo } from 'react';

import {
  avatarTrackNameForLocale,
  type TalkshowLocale,
} from '@/lib/talkshow/locale';

/** Agent-published DyStream video track (muted; audio stays on agent / locale audio). */
export function useAgentAvatarVideo(
  agentParticipant: Participant | undefined,
  viewerLocale: TalkshowLocale = 'en',
): TrackReference | undefined {
  const cameraTracks = useTracks([Track.Source.Camera], {
    onlySubscribed: true,
  });

  const wantedName = avatarTrackNameForLocale(viewerLocale);

  return useMemo(() => {
    if (!agentParticipant) return undefined;

    const named = cameraTracks.find(
      (ref) =>
        ref.participant.sid === agentParticipant.sid &&
        ref.publication?.trackName === wantedName &&
        ref.publication?.track,
    );
    if (named) return named;

    // Fallback: legacy talkshow-avatar or agent's only camera track
    const legacy = cameraTracks.find(
      (ref) =>
        ref.participant.sid === agentParticipant.sid &&
        ref.publication?.trackName === 'talkshow-avatar' &&
        ref.publication?.track,
    );
    if (legacy) return legacy;

    const anyCam = cameraTracks.find(
      (ref) =>
        ref.participant.sid === agentParticipant.sid && ref.publication?.track,
    );
    return anyCam;
  }, [cameraTracks, agentParticipant, wantedName]);
}

/** Backchannel track: a listener's face while their backchannel plays (exact name only —
 * falling back to another camera track would show the speaker in the listener's tile). */
export function useBackchannelAvatarVideo(
  agentParticipant: Participant | undefined,
): TrackReference | undefined {
  const cameraTracks = useTracks([Track.Source.Camera], { onlySubscribed: true });
  return useMemo(() => {
    if (!agentParticipant) return undefined;
    return cameraTracks.find(
      (ref) =>
        ref.participant.sid === agentParticipant.sid &&
        ref.publication?.trackName === 'talkshow-avatar-bc' &&
        ref.publication?.track,
    );
  }, [cameraTracks, agentParticipant]);
}
