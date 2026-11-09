import { describe, expect, it } from 'vitest';

import { joinSegments, splitAtBackchannels } from './transcriptParts';

const bc = (id: string, at: number) => ({ id, at, role: 'guest', name: 'Amy', text: 'mm-hmm' });

describe('joinSegments', () => {
  it('joins TTS chunk segments with one space, in arrival order', () => {
    const segments = new Map([
      ['s1', 'Reading a correct answer '],
      ['s2', ' is not learning.'],
    ]);
    expect(joinSegments(segments)).toBe('Reading a correct answer is not learning.');
  });

  it('ignores segments with no words yet', () => {
    expect(joinSegments(new Map([['s1', '  ']]))).toBe('');
  });
});

describe('splitAtBackchannels', () => {
  it('places each backchannel where it was heard', () => {
    const parts = splitAtBackchannels('one two three', [bc('b', 8), bc('a', 4)]);
    expect(parts.map((p) => (p.kind === 'text' ? p.text : `[${p.item.id}]`))).toEqual([
      'one ',
      '[a]',
      'two ',
      '[b]',
      'three',
    ]);
  });

  it('clamps an anchor past the shown text to its end', () => {
    const parts = splitAtBackchannels('one', [bc('a', 10)]);
    expect(parts.map((p) => p.kind)).toEqual(['text', 'backchannel']);
  });

  it('keeps a backchannel at the very start before any words', () => {
    const parts = splitAtBackchannels('', [bc('a', 0)]);
    expect(parts.map((p) => p.kind)).toEqual(['backchannel']);
  });
});
