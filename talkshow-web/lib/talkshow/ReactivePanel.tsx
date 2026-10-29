'use client';

import { useDataChannel, useRoomContext } from '@livekit/components-react';
import { useState } from 'react';
import { publishDemoStop, publishHumanInterrupt, publishReactiveDemo } from './control';
import { UI_TOPIC } from './roles';
import { readDecision, RULE_LABELS, type DecisionRecord, type DemoMode } from './reactive';
import styles from '@/styles/TalkshowStage.module.css';

export function ReactivePanel({
  canStart,
  onStart,
  aiOnly,
  running,
}: {
  canStart: boolean;
  onStart: () => void;
  aiOnly?: boolean;
  running?: boolean;
}) {
  const room = useRoomContext();
  const [records, setRecords] = useState<DecisionRecord[]>([]);
  const [active, setActive] = useState(false);
  const [speaker, setSpeaker] = useState('');
  const [heard, setHeard] = useState('');
  const [mode, setMode] = useState<DemoMode>('live');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [stopMs, setStopMs] = useState<number | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const jsonl = records.map((record) => JSON.stringify(record)).join('\n') + '\n';

  useDataChannel(UI_TOPIC, (msg) => {
    let event: Record<string, unknown>;
    try {
      event = JSON.parse(new TextDecoder().decode(msg.payload));
    } catch {
      return;
    }
    if (!event || typeof event !== 'object') return;
    const record = readDecision(event);
    if (record) setRecords((prev) => [...prev.slice(-99), record]);
    if (event.type === 'reactive_reset') {
      setRecords([]);
      setHeard('');
      setSpeaker('Amy');
      setError('');
      setStopMs(null);
      setExportOpen(false);
      setCopied(false);
      setPending(true);
      setMode(event.mode as DemoMode);
    } else if (event.type === 'reactive_generating') {
      setPending(true);
      setActive(false);
      setSpeaker(String(event.name ?? ''));
      setHeard('');
    } else if (event.type === 'reactive_state') {
      setActive(event.active === true);
      setSpeaker(String(event.name ?? ''));
      setHeard(String(event.heard ?? ''));
      setPending(false);
      if (typeof event.stop_ack_ms === 'number') setStopMs(event.stop_ack_ms);
    } else if (event.type === 'reactive_complete') {
      setPending(false);
      setActive(false);
    } else if (event.type === 'reactive_error') {
      setError(String(event.message ?? 'Could not start.'));
      setPending(false);
    }
  });

  async function start(next: DemoMode) {
    setPending(true);
    setError('');
    try {
      await publishReactiveDemo(room, next);
      onStart();
    } catch {
      setPending(false);
      setError('Could not send. Try again.');
    }
  }

  function exportDecisions() {
    const blob = new Blob([jsonl], {
      type: 'application/x-ndjson',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'floor-decisions.jsonl';
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function copyDecisions() {
    try {
      await navigator.clipboard.writeText(jsonl);
      setCopied(true);
    } catch {
      setError('Copy is unavailable. Select the JSONL text or use Download.');
    }
  }

  return (
    <aside className={styles.reactivePanel} aria-label="Floor decisions">
      <div className={styles.reactiveHeading}>
        <h3>Floor control</h3>
        <button
          type="button"
          onClick={() => {
            setExportOpen(!exportOpen);
            setCopied(false);
          }}
          disabled={records.length === 0}
          aria-expanded={exportOpen}
        >
          Export
        </button>
      </div>
      <div className={styles.reactiveNow}>
        <span>
          {pending ? `Generating ${speaker || 'Amy'}…` : active ? `${speaker} speaking` : 'Ready'}
        </span>
        <button
          type="button"
          className={styles.interruptButton}
          disabled={aiOnly ? !running : !active && !pending}
          onClick={() =>
            void (aiOnly ? publishDemoStop(room) : publishHumanInterrupt(room)).catch(() =>
              setError('Could not stop.'),
            )
          }
        >
          {aiOnly ? 'Stop' : 'Interrupt'}
        </button>
      </div>
      <div className={styles.reactiveContent}>
        {exportOpen && (
          <div className={styles.reactiveExport}>
            <textarea aria-label="Decision JSONL" readOnly value={jsonl} rows={4} />
            <button type="button" onClick={() => void copyDecisions()}>
              {copied ? 'Copied' : 'Copy JSONL'}
            </button>{' '}
            <button type="button" onClick={exportDecisions}>
              Download
            </button>
          </div>
        )}
        <div className={styles.reactiveModes} aria-label="Short demos">
          <button type="button" disabled={!canStart || pending} onClick={() => void start('live')}>
            {aiOnly ? 'Start demo' : 'AI interrupt'}
          </button>
          {!aiOnly && (
            <button
              type="button"
              disabled={!canStart || pending}
              onClick={() => void start('queue')}
            >
              Queue only
            </button>
          )}
          {!aiOnly && (
            <button
              type="button"
              disabled={!canStart || pending}
              onClick={() => void start('baseline')}
            >
              Sequential
            </button>
          )}
        </div>
        {!aiOnly && (
          <p className={styles.reactiveNote}>AI homework · preset stance · live generation</p>
        )}
        {heard && !aiOnly && <p className={styles.reactiveHeard}>{heard}</p>}
        {stopMs !== null && (
          <p className={styles.reactiveNote}>
            Stop ack {stopMs.toFixed(0)} ms · backend, not browser audio
          </p>
        )}
        {error && <p role="alert">{error}</p>}
        <div className={styles.reactiveDecisions} aria-live="polite">
          {records.length === 0 && (
            <p className={styles.muted}>
              {mode === 'baseline'
                ? 'Sequential generation. No AI interruption requests.'
                : 'Waiting for a request…'}
            </p>
          )}
          {records.slice(-6).map((record, i) => (
            <article
              key={`${record.ts}-${i}`}
              data-status={record.status}
              className={styles.reactiveDecision}
            >
              <header>
                <strong>{record.name}</strong>
                <span>{record.status}</span>
              </header>
              <p>{RULE_LABELS[record.rule] ?? record.rule}</p>
              <details>
                <summary>
                  {record.source === 'llm' ? 'LLM request' : aiOnly ? 'Controller' : 'Rule / human'}
                </summary>
                <p>{record.reason}</p>
                {record.evidence && <blockquote>{record.evidence}</blockquote>}
                <small>
                  {record.rule} {record.epoch !== undefined && `· turn ${record.epoch}`}
                </small>
              </details>
            </article>
          ))}
        </div>
      </div>
    </aside>
  );
}
