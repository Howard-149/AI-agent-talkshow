'use client';

/**
 * Per-panelist emotion readout: spoken label, last appraisal feeling, and P/A/D bars.
 */
import styles from '@/styles/TalkshowStage.module.css';

export type PadState = {
  pad: [number, number, number];
  emotion: string;
  feeling?: string;
};

const AXES = ['P', 'A', 'D'] as const;
const AXIS_TITLES = { P: 'Pleasure', A: 'Arousal', D: 'Dominance' } as const;

function titleCase(label: string): string {
  return label.charAt(0).toUpperCase() + label.slice(1);
}

export function PadMeter({ state }: { state?: PadState }) {
  const pad = state?.pad ?? [0, 0, 0];
  return (
    <div className={styles.padMeter}>
      <div className={styles.padEmotion}>
        <span className={styles.padLabel}>{titleCase(state?.emotion || 'neutral')}</span>
        {state?.feeling ? <span className={styles.padFeeling}>“{state.feeling}”</span> : null}
      </div>
      {AXES.map((axis, i) => {
        const v = Math.max(-1, Math.min(1, pad[i] ?? 0));
        const width = `${Math.abs(v) * 50}%`;
        return (
          <div key={axis} className={styles.padRow} title={AXIS_TITLES[axis]}>
            <span className={styles.padAxis}>{axis}</span>
            <span className={styles.padTrack}>
              <span
                className={styles.padFill}
                data-sign={v < 0 ? 'neg' : 'pos'}
                style={v < 0 ? { right: '50%', width } : { left: '50%', width }}
              />
            </span>
            <span className={styles.padValue}>
              {v >= 0 ? '+' : '−'}
              {Math.abs(v).toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
