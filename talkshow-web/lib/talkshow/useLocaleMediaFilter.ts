'use client';

/**
 * Subscribes only to agent audio and avatar tracks that match the viewer's preferred locale.
 */
import { useRoomContext } from '@livekit/components-react';
import {
  RoomEvent,
  Track,
  type RemoteParticipant,
  type RemoteTrackPublication,
} from 'livekit-client';
import { useEffect } from 'react';

import {
  audioTrackNameForLocale,
  avatarTrackNameForLocale,
  type TalkshowLocale,
} from '@/lib/talkshow/locale';
import { parseAgentMetadata } from '@/lib/talkshow/roles';

/**
 * Demand-driven media:
 * - Single locale: agent mic + talkshow-avatar (or talkshow-avatar-zh when primary is zh).
 * - Mixed: en keeps default mic + talkshow-avatar; zh uses talkshow-audio-zh + talkshow-avatar-zh.
 */
export function useLocaleMediaFilter(viewerLocale: TalkshowLocale): void {
  const room = useRoomContext();

  useEffect(() => {
    const wantedAvatar = avatarTrackNameForLocale(viewerLocale);
    const namedAudio = audioTrackNameForLocale(viewerLocale);

    const agentHasNamedAudio = (): boolean => {
      for (const p of room.remoteParticipants.values()) {
        const meta = parseAgentMetadata(p.metadata);
        if (!meta?.talkshowAgent) continue;
        for (const pub of p.trackPublications.values()) {
          if (pub.kind === Track.Kind.Audio && pub.trackName === namedAudio) {
            return true;
          }
        }
      }
      return false;
    };

    const syncPublication = (
      participant: RemoteParticipant,
      publication: RemoteTrackPublication,
    ) => {
      const meta = parseAgentMetadata(participant.metadata);
      if (!meta?.talkshowAgent) return;

      const name = publication.trackName || '';
      let want = true;

      if (publication.kind === Track.Kind.Video) {
        if (name.startsWith('talkshow-avatar')) {
          want = name === wantedAvatar;
          if (!want && viewerLocale === 'zh' && name === 'talkshow-avatar') {
            const hasZh = [...room.remoteParticipants.values()].some((p) =>
              [...p.trackPublications.values()].some(
                (pub) => pub.trackName === 'talkshow-avatar-zh',
              ),
            );
            want = !hasZh;
          }
        }
      } else if (publication.kind === Track.Kind.Audio) {
        if (name.startsWith('talkshow-audio-')) {
          want = name === namedAudio;
        } else {
          want = !agentHasNamedAudio();
        }
      }

      if (publication.isSubscribed !== want) {
        try {
          publication.setSubscribed(want);
        } catch (err) {
          console.warn('locale media subscribe failed', name, err);
        }
      }
    };

    const refreshAll = () => {
      for (const p of room.remoteParticipants.values()) {
        p.trackPublications.forEach((pub) => {
          syncPublication(p, pub);
        });
      }
    };

    const onPublished = (
      publication: RemoteTrackPublication,
      participant: RemoteParticipant,
    ) => {
      syncPublication(participant, publication);
      refreshAll();
    };

    room.on(RoomEvent.TrackPublished, onPublished);
    room.on(RoomEvent.TrackUnpublished, refreshAll);
    room.on(RoomEvent.ParticipantConnected, refreshAll);

    refreshAll();

    return () => {
      room.off(RoomEvent.TrackPublished, onPublished);
      room.off(RoomEvent.TrackUnpublished, refreshAll);
      room.off(RoomEvent.ParticipantConnected, refreshAll);
    };
  }, [room, viewerLocale]);
}
