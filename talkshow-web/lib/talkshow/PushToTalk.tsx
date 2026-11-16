'use client';

/**
 * Push-to-talk for noisy rooms: while on, the mic stays muted except while the visitor holds
 * Space or the "Hold to talk" button. Any sound on an open mic (a cough, the crowd, the show's
 * own voice from the speakers) cuts the current line, so a demo floor is safer with this on.
 */
import { useLocalParticipant } from '@livekit/components-react';
import { Track } from 'livekit-client';
import { useCallback, useEffect, useRef, useState } from 'react';

const PTT_KEY = 'talkshow.pushToTalk';

function micTrack(participant: ReturnType<typeof useLocalParticipant>['localParticipant']) {
  return participant.getTrackPublication(Track.Source.Microphone)?.audioTrack;
}

function typingInField(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  return !!el && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName));
}

export function PushToTalk() {
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant();
  const [enabled, setEnabled] = useState(false);
  const [talking, setTalking] = useState(false);
  const talkingRef = useRef(false);

  useEffect(() => {
    try {
      setEnabled(window.localStorage.getItem(PTT_KEY) === '1');
    } catch {
      /* ignore */
    }
  }, []);

  const setTalk = useCallback(
    (on: boolean) => {
      talkingRef.current = on;
      setTalking(on);
      const track = micTrack(localParticipant);
      if (!track) return;
      void (on ? track.unmute() : track.mute());
    },
    [localParticipant],
  );

  // On: keep the mic muted until held (also after the control bar re-enables it).
  // Switched off: unmute once, then the control bar owns the mic again.
  const wasEnabled = useRef(false);
  useEffect(() => {
    const track = micTrack(localParticipant);
    if (enabled && track && !talkingRef.current) void track.mute();
    if (!enabled && wasEnabled.current && track) void track.unmute();
    wasEnabled.current = enabled;
  }, [enabled, localParticipant, isMicrophoneEnabled]);

  useEffect(() => {
    if (!enabled) return;
    const down = (ev: KeyboardEvent) => {
      if (ev.code !== 'Space' || ev.repeat || typingInField(ev.target)) return;
      ev.preventDefault();
      setTalk(true);
    };
    const up = (ev: KeyboardEvent) => {
      if (ev.code !== 'Space' || typingInField(ev.target)) return;
      ev.preventDefault();
      setTalk(false);
    };
    const release = () => talkingRef.current && setTalk(false);
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    window.addEventListener('blur', release);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
      window.removeEventListener('blur', release);
    };
  }, [enabled, setTalk]);

  const toggle = () => {
    const next = !enabled;
    setEnabled(next);
    if (talkingRef.current) setTalk(false);
    try {
      window.localStorage.setItem(PTT_KEY, next ? '1' : '0');
    } catch {
      /* ignore */
    }
  };

  return (
    <>
      <button type="button" className="lk-button" aria-pressed={enabled} onClick={toggle}>
        Push to talk: {enabled ? 'on' : 'off'}
      </button>
      {enabled && (
        <button
          type="button"
          className="lk-button"
          aria-pressed={talking}
          style={talking ? { background: '#c0392b', color: '#fff' } : undefined}
          onPointerDown={(ev) => {
            ev.currentTarget.setPointerCapture(ev.pointerId);
            setTalk(true);
          }}
          onPointerUp={() => setTalk(false)}
          onPointerCancel={() => setTalk(false)}
        >
          {talking ? 'Talking…' : 'Hold to talk (Space)'}
        </button>
      )}
    </>
  );
}
