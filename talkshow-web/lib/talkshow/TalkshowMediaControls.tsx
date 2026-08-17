'use client';

/**
 * LiveKit ControlBar plus talkshow extras: settings drawer, transcript toggle, and hand raise.
 */
import { ControlBar } from '@livekit/components-react';

import { CameraSettings } from '@/lib/CameraSettings';
import { MicrophoneSettings } from '@/lib/MicrophoneSettings';
import type { TalkshowLocale } from '@/lib/talkshow/locale';
import styles from '@/styles/TalkshowStage.module.css';

type Props = {
  showSettings: boolean;
  onToggleSettings: () => void;
  showTranscript: boolean;
  onToggleTranscript: () => void;
  transcriptCount: number;
  handRaised: boolean;
  onToggleHandRaise: () => void;
  viewerLocale: TalkshowLocale;
  onLocaleChange: (locale: TalkshowLocale) => void;
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
  viewerLocale,
  onLocaleChange,
}: Props) {
  return (
    <>
      {showSettings && (
        <div className={styles.settingsPanel} role="dialog" aria-label="Device settings">
          <div className={styles.localeSettings} role="group" aria-label="Language">
            <span className={styles.localeSettingsLabel}>Language</span>
            <div className={styles.localeToggle}>
              <button
                type="button"
                className={`${styles.localeBtn} ${viewerLocale === 'en' ? styles.localeBtnOn : ''}`}
                aria-pressed={viewerLocale === 'en'}
                onClick={() => onLocaleChange('en')}
              >
                English
              </button>
              <button
                type="button"
                className={`${styles.localeBtn} ${viewerLocale === 'zh' ? styles.localeBtnOn : ''}`}
                aria-pressed={viewerLocale === 'zh'}
                onClick={() => onLocaleChange('zh')}
              >
                中文
              </button>
            </div>
            <p className={styles.localeHint}>
              Applies to transcript and upcoming TTS. Already-spoken lines keep their language.
            </p>
          </div>
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
