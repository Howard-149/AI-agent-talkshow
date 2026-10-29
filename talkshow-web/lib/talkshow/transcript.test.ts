import { describe, expect, it } from 'vitest';
import { updateTranscript, type TranscriptLine } from './transcript';

const partial = (text: string): TranscriptLine => ({
  id: 'reactive-1',
  role: 'guest',
  speaker: 'Amy',
  text,
  ts: 1,
  final: false,
});

describe('playout-synchronized transcript', () => {
  it('updates a single turn progressively instead of appending duplicate lines', () => {
    let lines = updateTranscript([], partial('Students'));
    lines = updateTranscript(lines, partial('Students should use AI.'));
    lines = updateTranscript(lines, { ...partial('Students should use AI.'), final: true });
    expect(lines).toHaveLength(1);
    expect(lines[0].text).toBe('Students should use AI.');
    expect(lines[0].final).toBe(true);
  });
  it('keeps the audible partial fragment on interruption, but ignores late tails', () => {
    const heard = updateTranscript([], partial('First sentence. But I'));
    const closed = updateTranscript(heard, {
      ...partial('First sentence.'),
      final: true,
      interrupted: true,
    });
    expect(closed[0].text).toBe('First sentence. But I');
    expect(
      updateTranscript(closed, partial('First sentence. But I have more unspoken words')),
    ).toBe(closed);
  });
  it('preserves timestamps, speakers, and independent turn IDs', () => {
    const lines = updateTranscript([partial('Amy')], {
      ...partial('Ryan replies'),
      id: 'reactive-2',
      role: 'commentator',
      speaker: 'Ryan',
    });
    expect(lines.map((l) => l.speaker)).toEqual(['Amy', 'Ryan']);
    expect(updateTranscript(lines, { ...partial('Amy speaks'), ts: 99 })[0].ts).toBe(1);
  });
});
