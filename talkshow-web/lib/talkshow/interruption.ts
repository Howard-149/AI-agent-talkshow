import { readDecision } from './reactive';

export type SpeakingTurn = { role: string; name: string; epoch: number };
export type InterruptionNotice = {
  id: string;
  epoch: number;
  byRole: string;
  byName: string;
  targetRole: string;
  targetName: string;
};

const INTERRUPTION_RULES = new Set(['grounded_rebuttal', 'moderator_cut']);

/** Only a real, current-turn AI grant is an interruption, never a queued request. */
export function readInterruption(
  event: unknown,
  current: SpeakingTurn | null,
): InterruptionNotice | null {
  const record = readDecision(event);
  if (
    !current ||
    !record ||
    record.status !== 'granted' ||
    record.source !== 'llm' ||
    !INTERRUPTION_RULES.has(record.rule) ||
    record.epoch !== current.epoch ||
    record.role === current.role ||
    record.role === 'human'
  )
    return null;
  return {
    id: `interrupt-${record.epoch}-${record.role}`,
    epoch: current.epoch,
    byRole: record.role,
    byName: record.name,
    targetRole: current.role,
    targetName: current.name,
  };
}
