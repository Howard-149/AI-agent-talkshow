import type { Room } from 'livekit-client';

export const CONTROL_TOPIC = 'talkshow/control';

export type HandRaiseControl = {
  type: 'hand_raise';
  raised: boolean;
  topic?: string;
  reason?: string;
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
