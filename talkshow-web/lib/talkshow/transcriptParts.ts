/**
 * Pure helpers for transcript lines: synced-segment joining and backchannel placement.
 */

/** A listener's backchannel, anchored at a character offset of the line it overlapped. */
export type TranscriptBackchannel = {
  id: string;
  at: number;
  role: string;
  name: string;
  text: string;
};

export type LinePart =
  | { kind: 'text'; key: string; text: string }
  | { kind: 'backchannel'; key: string; item: TranscriptBackchannel };

/** A line's words split at its backchannels, each placed where it was heard. */
export function splitAtBackchannels(
  text: string,
  backchannels: TranscriptBackchannel[],
): LinePart[] {
  const parts: LinePart[] = [];
  let cursor = 0;
  for (const item of [...backchannels].sort((a, b) => a.at - b.at)) {
    const at = Math.min(Math.max(item.at, cursor), text.length);
    if (at > cursor) parts.push({ kind: 'text', key: `t-${item.id}`, text: text.slice(cursor, at) });
    parts.push({ kind: 'backchannel', key: item.id, item });
    cursor = at;
  }
  if (cursor < text.length) parts.push({ kind: 'text', key: 'rest', text: text.slice(cursor) });
  return parts;
}

/** Joins a line's synced segments (one per TTS chunk) in arrival order. */
export function joinSegments(segments: Map<string, string>): string {
  return [...segments.values()]
    .map((s) => s.trim())
    .filter(Boolean)
    .join(' ');
}
