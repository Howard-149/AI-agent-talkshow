'use client';

import { useEffect, useRef } from 'react';
import styles from '@/styles/TalkshowPanel.module.css';

export type TranscriptLine = {
  id: string;
  speaker: string;
  role: string;
  text: string;
  ts: number;
};

export function TranscriptPanel({ lines }: { lines: TranscriptLine[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lines.length]);

  return (
    <div className={styles.transcript}>
      <h3 className={styles.transcriptTitle}>Transcript</h3>
      <div className={styles.transcriptScroll}>
        {lines.length === 0 && <p className={styles.muted}>Waiting for speech…</p>}
        {lines.map((line) => (
          <article key={line.id} className={styles.line} data-role={line.role}>
            <header className={styles.lineHeader}>
              <span className={styles.speaker}>{line.speaker}</span>
              <time className={styles.time}>
                {new Date(line.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </time>
            </header>
            <p className={styles.lineText}>{line.text}</p>
          </article>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
