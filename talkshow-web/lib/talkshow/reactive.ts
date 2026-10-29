export type DemoMode = 'live' | 'queue' | 'baseline';
export type DecisionRecord = {
  type: 'reactive_decision';
  role: string;
  name: string;
  status: 'granted' | 'queued' | 'rejected' | 'none';
  rule: string;
  reason: string;
  source: string;
  evidence?: string;
  epoch?: number;
  ts: number;
};

export function readDecision(event: unknown): DecisionRecord | null {
  if (!event || typeof event !== 'object') return null;
  const obj = event as Record<string, unknown>;
  if (
    obj.type !== 'reactive_decision' ||
    typeof obj.role !== 'string' ||
    typeof obj.status !== 'string' ||
    !['granted', 'queued', 'rejected', 'none'].includes(obj.status)
  )
    return null;
  return {
    type: 'reactive_decision',
    role: obj.role,
    name: typeof obj.name === 'string' ? obj.name : obj.role,
    status: obj.status as DecisionRecord['status'],
    rule: typeof obj.rule === 'string' ? obj.rule : '',
    reason: typeof obj.reason === 'string' ? obj.reason : '',
    source: typeof obj.source === 'string' ? obj.source : '',
    evidence: typeof obj.evidence === 'string' ? obj.evidence : undefined,
    epoch: typeof obj.epoch === 'number' ? obj.epoch : undefined,
    ts: Date.now(),
  };
}

export const RULE_LABELS: Record<string, string> = {
  human_priority: 'Human first',
  superseded_by_human: 'Human takes priority over the AI request',
  grounded_rebuttal: 'Direct rebuttal',
  moderator_cut: 'Moderator steers back to the topic',
  moderator_waits: 'Moderator keeps listening',
  wait_for_turn_end: 'Wait for turn end',
  minimum_speaking_time: 'Let the speaker finish their opening',
  cooldown: 'Interruption cooldown',
  queue_only_or_chain_limit: 'Queue-only policy',
  stale_epoch: 'Old turn discarded',
  evidence_not_heard: 'Claim not in played speech',
  fresh_draft_at_turn_end: 'Refresh with latest context',
  model_error: 'Invalid request',
  no_new_point: 'No new point',
};
