/**
 * Client → agent control messages over LiveKit data channels (hand raise, viewer locale).
 */
import type { Room } from 'livekit-client';

import type { TalkshowLocale } from '@/lib/talkshow/locale';
import type { DemoMode } from './reactive';

export const CONTROL_TOPIC = 'talkshow/control';

export async function publishDemoStop(room: Room): Promise<void> {
  await room.localParticipant.publishData(
    new TextEncoder().encode(JSON.stringify({ type: 'reactive_stop' })),
    { reliable: true, topic: CONTROL_TOPIC },
  );
}

export async function publishHumanInterrupt(room: Room): Promise<void> {
  await room.localParticipant.publishData(
    new TextEncoder().encode(JSON.stringify({ type: 'human_interrupt' })),
    { reliable: true, topic: CONTROL_TOPIC },
  );
}

export async function publishReactiveDemo(room: Room, mode: DemoMode): Promise<void> {
  await room.localParticipant.publishData(
    new TextEncoder().encode(JSON.stringify({ type: 'reactive_demo', mode })),
    { reliable: true, topic: CONTROL_TOPIC },
  );
}

// This is the same text-turn protocol used by eval/ghost_session. The server
// checks participant metadata and grants before accepting a human turn.
export async function publishTextTurn(room: Room, text: string): Promise<void> {
  await room.localParticipant.publishData(
    new TextEncoder().encode(JSON.stringify({ type: 'ghost_human_turn', text })),
    { reliable: true, topic: CONTROL_TOPIC },
  );
}

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
  await room.localParticipant.publishData(new TextEncoder().encode(JSON.stringify(payload)), {
    reliable: true,
    topic: CONTROL_TOPIC,
  });
}

export async function publishSetLocale(room: Room, locale: TalkshowLocale): Promise<void> {
  const payload: SetLocaleControl = {
    type: 'set_locale',
    locale,
  };
  await room.localParticipant.publishData(new TextEncoder().encode(JSON.stringify(payload)), {
    reliable: true,
    topic: CONTROL_TOPIC,
  });
}
