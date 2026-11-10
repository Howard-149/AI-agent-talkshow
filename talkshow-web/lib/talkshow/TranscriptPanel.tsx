'use client';

/**
 * Scrollable live transcript: agent lines stream in step with their audio, interrupts are
 * flagged in amber, and backchannels sit inline at the point they were heard.
 */
import { Fragment, useEffect, useRef } from 'react';

import {
  splitAtBackchannels,
  type TranscriptBackchannel,
} from '@/lib/talkshow/transcriptParts';
import type { InterruptNotice, SyncedTranscriptLine } from '@/lib/talkshow/usePlayoutSyncedUi';
import styles from '@/styles/TalkshowTranscript.module.css';

export type TranscriptLine = SyncedTranscriptLine;

function Backchannel({ item }: { item: TranscriptBackchannel }) {
  return (
    <span className={styles.backchannel} data-role={item.role} title={`${item.name} backchannel`}>
      <span className={styles.backchannelName}>{item.name}</span>
      {item.text}
    </span>
  );
}

/** The line's words with its backchannels inserted where they were heard. */
function LineText({ line }: { line: SyncedTranscriptLine }) {
  return (
    <p className={styles.lineText}>
      {splitAtBackchannels(line.text, line.backchannels).map((part) =>
        part.kind === 'text' ? (
          <Fragment key={part.key}>{part.text}</Fragment>
        ) : (
          <Backchannel key={part.key} item={part.item} />
        ),
      )}
      {line.streaming && <span className={styles.caret} aria-hidden />}
    </p>
  );
}

export function TranscriptPanel({
  lines,
  interruptNotice = null,
}: {
  lines: SyncedTranscriptLine[];
  interruptNotice?: InterruptNotice | null;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const last = lines[lines.length - 1];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [lines.length, last?.text.length, last?.backchannels.length]);

  return (
    <div className={styles.transcript}>
      <header className={styles.header}>
        <h3 className={styles.title}>Transcript</h3>
        {interruptNotice && (
          <div key={interruptNotice.id} className={styles.notice} role="status">
            <span className={styles.noticeTag}>AI interruption</span>
            <span className={styles.noticeText}>
              {interruptNotice.byName} interrupts {interruptNotice.name}
            </span>
          </div>
        )}
      </header>
      <div className={styles.scroll}>
        {lines.length === 0 && <p className={styles.muted}>Waiting for speech…</p>}
        {lines.map((line) => (
          <article
            key={line.id}
            className={styles.line}
            data-role={line.role}
            data-interrupted={line.interruptedBy ? 'true' : undefined}
          >
            <header className={styles.lineHeader}>
              <span className={styles.who}>
                <span className={styles.speaker}>{line.speaker}</span>
                {line.role !== 'human' && <span className={styles.aiTag}>AI</span>}
                {line.interruptedBy && (
                  <span className={styles.flag}>Interrupted by {line.interruptedBy}</span>
                )}
                {line.tookFloor && <span className={styles.flag}>Took the floor</span>}
              </span>
              <time className={styles.time}>
                {new Date(line.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </time>
            </header>
            <LineText line={line} />
          </article>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
