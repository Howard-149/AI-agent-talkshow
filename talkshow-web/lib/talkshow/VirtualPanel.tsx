'use client';

import type { PanelistDef, PanelRole } from '@/lib/talkshow/roles';
import { PanelAvatar } from '@/lib/talkshow/PanelAvatar';
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
  lipSyncActive?: boolean;
  humans: HumanSeat[];
  waitingForAgent: boolean;
};

export function VirtualPanel({
  panelists,
  activeRole,
  lipSyncActive = false,
  humans,
  waitingForAgent,
}: Props) {
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
          <PanelAvatar
            panelist={p}
            isSpeaking={activeRole === p.role && lipSyncActive}
          />
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
