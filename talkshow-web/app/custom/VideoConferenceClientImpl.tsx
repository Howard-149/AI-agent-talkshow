'use client';

import { TalkshowRoom } from '@/lib/talkshow/TalkshowRoom';

/** Token-tab entry: talkshow stage (no Meet participant grid). */
export function VideoConferenceClientImpl(props: {
  liveKitUrl: string;
  token: string;
  codec?: unknown;
  singlePeerConnection: boolean | undefined;
}) {
  return (
    <TalkshowRoom
      liveKitUrl={props.liveKitUrl}
      token={props.token}
      singlePeerConnection={props.singlePeerConnection}
    />
  );
}
