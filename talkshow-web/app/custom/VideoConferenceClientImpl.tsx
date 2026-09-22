'use client';

import { TalkshowRoom } from '@/lib/talkshow/TalkshowRoom';
import type { TalkshowLocale } from '@/lib/talkshow/locale';

/** Token-tab entry: talkshow stage (no Meet participant grid). */
export function VideoConferenceClientImpl(props: {
  liveKitUrl: string;
  token: string;
  codec?: unknown;
  singlePeerConnection: boolean | undefined;
  locale?: TalkshowLocale;
  recordMode?: boolean;
}) {
  return (
    <TalkshowRoom
      liveKitUrl={props.liveKitUrl}
      token={props.token}
      singlePeerConnection={props.singlePeerConnection}
      locale={props.locale}
      recordMode={props.recordMode}
    />
  );
}
