/** Cumulative playback captions replace a single turn; late packets cannot reopen it. */
export type TranscriptLine = {
  id: string;
  role: string;
  speaker: string;
  text: string;
  ts: number;
  final: boolean;
  interrupted?: boolean;
};

export function updateTranscript(
  lines: TranscriptLine[],
  update: TranscriptLine,
): TranscriptLine[] {
  const index = lines.findIndex((line) => line.id === update.id);
  if (index < 0) return update.text.trim() ? [...lines, update] : lines;
  const old = lines[index];
  if (old.final && !update.final) return lines;
  const text =
    update.interrupted && old.text.startsWith(update.text.trim()) ? old.text : update.text;
  return lines.map((line, i) => (i === index ? { ...update, text, ts: old.ts } : line));
}
