/**
 * Client → agent control messages over LiveKit data channels (hand raise, viewer locale).
 */
import type { Room } from 'livekit-client';

import type { TalkshowLocale } from '@/lib/talkshow/locale';

export const CONTROL_TOPIC = 'talkshow/control';

export type HandRaiseControl = {
  type: 'hand_raise';
  raised: boolean;
  topic?: string;
  reason?: string;
};

export type SetLocaleControl = {
  type: 'set_locale';
  locale: TalkshowLocale;
};

export async function publishHandRaise(
  room: Room,
  raised: boolean,
  opts?: { topic?: string; reason?: string },
): Promise<void> {
  const payload: HandRaiseControl = {
    type: 'hand_raise',
    raised,
    topic: opts?.topic,
    reason: opts?.reason,
  };
  await room.localParticipant.publishData(
    new TextEncoder().encode(JSON.stringify(payload)),
    { reliable: true, topic: CONTROL_TOPIC },
  );
}

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
