'use client';

import styles from '@/styles/TalkshowStage.module.css';

type Props = {
  roomName: string | null;
  message: string;
  onRetry: () => void;
  onLeave: () => void;
};

export function ConnectionRecovery({ roomName, message, onRetry, onLeave }: Props) {
  const resetRoom = async () => {
    if (!roomName) return;
    const res = await fetch(
      `/api/room/reset?roomName=${encodeURIComponent(roomName)}`,
      { method: 'POST' },
    );
    if (!res.ok) {
      const text = await res.text();
      alert(`Room reset failed: ${text}`);
      return;
    }
    onRetry();
  };

  return (
    <div className={styles.recoveryBanner} role="alert">
      <p className={styles.recoveryTitle}>Connection problem</p>
      <p className={styles.recoveryText}>{message}</p>
      <ul className={styles.recoveryList}>
        <li>Wait ~30s for LiveKit to drop a ghost session after closing the tab.</li>
        <li>Mint a fresh token: <code>python api/tokens.py --room talkshow-dev --identity you</code></li>
        {roomName && (
          <li>
            Or reset room <strong>{roomName}</strong> (kicks agent + all participants).
          </li>
        )}
      </ul>
      <div className={styles.recoveryActions}>
        <button type="button" className="lk-button" onClick={onRetry}>
          Retry connect
        </button>
        {roomName && (
          <button type="button" className="lk-button" onClick={() => void resetRoom()}>
            Reset room (dev)
          </button>
        )}
        <button type="button" className="lk-button" onClick={onLeave}>
          Back to connect form
        </button>
      </div>
    </div>
  );
}
