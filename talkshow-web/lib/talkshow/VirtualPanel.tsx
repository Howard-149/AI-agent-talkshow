'use client';

import type { PanelistDef, PanelRole } from '@/lib/talkshow/roles';
import styles from '@/styles/TalkshowPanel.module.css';

export type HumanSeat = {
  key: string;
  name: string;
  label: string;
  isLocal?: boolean;
};

type Props = {
  panelists: PanelistDef[];
  activeRole: PanelRole | null;
  humans: HumanSeat[];
  waitingForAgent: boolean;
};

export function VirtualPanel({ panelists, activeRole, humans, waitingForAgent }: Props) {
  return (
    <div className={styles.panelGrid}>
      {humans.map((h) => (
        <div
          key={h.key}
          className={styles.card}
          data-human
          data-speaking={activeRole === 'human' && h.isLocal ? 'true' : 'false'}
        >
          <div className={styles.avatarHuman}>{h.isLocal ? 'You' : h.name[0] ?? '?'}</div>
          <div className={styles.meta}>
            <strong>{h.name}</strong>
            <span>{h.label}</span>
          </div>
        </div>
      ))}
      {panelists.map((p) => (
        <div
          key={p.role}
          className={styles.card}
          data-speaking={activeRole === p.role ? 'true' : 'false'}
          style={{ '--role-color': p.color } as React.CSSProperties}
        >
          <div className={styles.avatar} style={{ background: p.color }}>
            {p.name[0]}
          </div>
          <div className={styles.meta}>
            <strong>{p.name}</strong>
            <span>{p.label}</span>
          </div>
        </div>
      ))}
      {waitingForAgent && panelists.length === 0 && (
        <p className={styles.waiting}>Waiting for talkshow agent…</p>
      )}
    </div>
  );
}
