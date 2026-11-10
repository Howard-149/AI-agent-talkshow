/**
 * Client → agent control messages over LiveKit data channels (viewer locale).
 */
import type { Room } from 'livekit-client';

import type { TalkshowLocale } from '@/lib/talkshow/locale';

export const CONTROL_TOPIC = 'talkshow/control';

export type SetLocaleControl = {
  type: 'set_locale';
  locale: TalkshowLocale;
};

export async function publishSetLocale(
  room: Room,
  locale: TalkshowLocale,
): Promise<void> {
  const payload: SetLocaleControl = {
    type: 'set_locale',
    locale,
  };
  await room.localParticipant.publishData(
    new TextEncoder().encode(JSON.stringify(payload)),
    { reliable: true, topic: CONTROL_TOPIC },
  );
}
