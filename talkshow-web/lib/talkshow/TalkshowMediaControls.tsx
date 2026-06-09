'use client';

import { ControlBar } from '@livekit/components-react';

import { CameraSettings } from '@/lib/CameraSettings';
import { MicrophoneSettings } from '@/lib/MicrophoneSettings';
import styles from '@/styles/TalkshowStage.module.css';

type Props = {
  showSettings: boolean;
  onToggleSettings: () => void;
  showTranscript: boolean;
  onToggleTranscript: () => void;
  transcriptCount: number;
  handRaised: boolean;
  onToggleHandRaise: () => void;
};

/** Meet ControlBar + talkshow extras (settings drawer, transcript). */
export function TalkshowMediaControls({
  showSettings,
  onToggleSettings,
  showTranscript,
  onToggleTranscript,
  transcriptCount,
  handRaised,
  onToggleHandRaise,
}: Props) {
  return (
    <>
      {showSettings && (
        <div className={styles.settingsPanel} role="dialog" aria-label="Device settings">
          <MicrophoneSettings />
          <CameraSettings />
        </div>
      )}

      <div className={styles.controlBarRow}>
        <ControlBar
          className={styles.lkBar}
          controls={{
            microphone: true,
            camera: true,
            screenShare: false,
            chat: false,
            leave: true,
            settings: false,
          }}
          variation="verbose"
        />

        <div className={styles.extraControls}>
          <button
            type="button"
            className={`${styles.handToggle} ${handRaised ? styles.handToggleOn : ''}`}
            aria-pressed={handRaised}
            onClick={onToggleHandRaise}
            title="Raise hand for the floor"
          >
            {handRaised ? '✋ Hand up' : '✋ Raise hand'}
          </button>
          <button
            type="button"
            className={`${styles.transcriptToggle} ${showTranscript ? styles.transcriptToggleOn : ''}`}
            aria-pressed={showTranscript}
            onClick={onToggleTranscript}
          >
            Transcript
            {!showTranscript && transcriptCount > 0 && (
              <span className={styles.transcriptBadge}>{transcriptCount}</span>
            )}
          </button>
          <button
            type="button"
            className="lk-button"
            onClick={onToggleSettings}
            aria-pressed={showSettings}
          >
            Settings
          </button>
        </div>
      </div>
    </>
  );
}
