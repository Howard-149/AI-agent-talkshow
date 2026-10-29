import { describe, expect, it } from 'vitest';
import { readInterruption } from './interruption';

const current = { role: 'guest', name: 'Amy', epoch: 4 };
const grant = {
  type: 'reactive_decision',
  role: 'commentator',
  name: 'Ryan',
  epoch: 4,
  source: 'llm',
  status: 'granted',
  rule: 'grounded_rebuttal',
};

describe('visible AI interruption', () => {
  it('identifies the interrupter and the interrupted person from a real grant', () => {
    expect(readInterruption(grant, current)).toMatchObject({
      byName: 'Ryan',
      targetName: 'Amy',
      epoch: 4,
    });
  });
  it('treats a granted moderator cut as an interruption by the host', () => {
    expect(
      readInterruption({ ...grant, role: 'host', name: 'Lessac', rule: 'moderator_cut' }, current),
    ).toMatchObject({ byRole: 'host', byName: 'Lessac', targetName: 'Amy' });
  });
  it('does not label normal, queued, rejected, or stale requests as interruptions', () => {
    for (const changes of [
      { status: 'queued' },
      { status: 'rejected' },
      { status: 'none' },
      { epoch: 3 },
      { rule: 'wait_for_turn_end' },
      { role: 'guest' },
      { source: 'rule' },
      { role: 'human', source: 'human' },
    ]) {
      expect(readInterruption({ ...grant, ...changes }, current)).toBeNull();
    }
    expect(readInterruption(grant, null)).toBeNull();
  });
});
