import { describe, expect, it } from 'vitest';
import { readDecision } from './reactive';

describe('floor decision events', () => {
  it('accepts and preserves an explainable grounded decision', () => {
    const result = readDecision({
      type: 'reactive_decision',
      role: 'commentator',
      name: 'Ryan',
      status: 'granted',
      rule: 'grounded_rebuttal',
      reason: 'Learning needs practice',
      evidence: 'AI answers are enough',
      source: 'llm',
      epoch: 7,
    });
    expect(result?.status).toBe('granted');
    expect(result?.epoch).toBe(7);
    expect(result?.evidence).toBe('AI answers are enough');
  });
  it('fails closed for unsupported statuses', () => {
    expect(
      readDecision({ type: 'reactive_decision', role: 'guest', status: 'approved' }),
    ).toBeNull();
  });
  it('ignores non-decision or malformed messages', () => {
    for (const value of [null, [], 'x', { type: 'reactive_state' }])
      expect(readDecision(value)).toBeNull();
  });
});
