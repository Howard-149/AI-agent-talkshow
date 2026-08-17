'use client';

/**
 * LiveKit Room connection shell for the talkshow page (E2EE, reconnect, viewer locale).
 */
import { RoomContext, StartMediaButton } from '@livekit/components-react';
import {
  ExternalE2EEKeyProvider,
  LogLevel,
  Room,
  RoomConnectOptions,
  RoomEvent,
  RoomOptions,
  VideoPresets,
} from 'livekit-client';
import { useRouter } from 'next/navigation';
import { useEffect, useMemo, useRef, useState } from 'react';

import { DebugMode } from '@/lib/Debug';
import { ConnectionRecovery } from '@/lib/talkshow/ConnectionRecovery';
import {
  normalizeTalkshowLocale,
  localeFromAccessToken,
  readStoredViewerLocale,
  writeStoredViewerLocale,
  type TalkshowLocale,
} from '@/lib/talkshow/locale';
import { roomNameFromAccessToken } from '@/lib/talkshow/roomNameFromToken';
import { TalkshowView } from '@/lib/talkshow/TalkshowView';
import { useSetupE2EE } from '@/lib/useSetupE2EE';
import styles from '@/styles/TalkshowStage.module.css';

export type TalkshowConnectChoices = {
  /** Default false — opt in via controls to avoid ambient noise → false VAD turns */
  audioEnabled?: boolean;
  videoEnabled?: boolean;
  audioDeviceId?: string;
  videoDeviceId?: string;
};

export function TalkshowRoom(props: {
  liveKitUrl: string;
  token: string;
  e2eePassphrase?: string;
  singlePeerConnection?: boolean;
  userChoices?: TalkshowConnectChoices;
  locale?: TalkshowLocale;
}) {
  const keyProvider = useMemo(() => new ExternalE2EEKeyProvider(), []);
  const { worker, e2eePassphrase } = useSetupE2EE();
  const passphrase = props.e2eePassphrase ?? e2eePassphrase;
  const e2eeEnabled = !!(passphrase && worker);

  const viewerLocale = useMemo((): TalkshowLocale => {
    // Join-token metadata is what the agent uses for opening TTS — prefer it over UI/localStorage.
    const fromToken = localeFromAccessToken(props.token);
    if (fromToken) return fromToken;
    if (props.locale) return normalizeTalkshowLocale(props.locale);
    return readStoredViewerLocale();
  }, [props.locale, props.token]);

  useEffect(() => {
    writeStoredViewerLocale(viewerLocale);
  }, [viewerLocale]);

  const [e2eeSetupComplete, setE2eeSetupComplete] = useState(!e2eeEnabled);
  const [connectError, setConnectError] = useState<string | null>(null);
  const connectedRef = useRef(false);
  const skipDisconnectRedirectRef = useRef(false);
  const router = useRouter();
  const roomName = useMemo(() => roomNameFromAccessToken(props.token), [props.token]);

  const roomOptions = useMemo((): RoomOptions => {
    const choices = props.userChoices;
    return {
      audioCaptureDefaults: {
        deviceId: choices?.audioDeviceId,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      videoCaptureDefaults: {
        deviceId: choices?.videoDeviceId,
        resolution: VideoPresets.h720,
      },
      publishDefaults: {
        dtx: false,
        red: !e2eeEnabled,
      },
      adaptiveStream: true,
      dynacast: true,
      e2ee: e2eeEnabled
        ? {
            keyProvider,
            worker: worker!,
          }
        : undefined,
      singlePeerConnection: props.singlePeerConnection,
    };
  }, [e2eeEnabled, keyProvider, worker, props.singlePeerConnection, props.userChoices]);

  const room = useMemo(() => new Room(roomOptions), [roomOptions]);

  const connectOptions = useMemo((): RoomConnectOptions => {
    return { autoSubscribe: true };
  }, []);

  useEffect(() => {
    if (!e2eeEnabled || !passphrase) {
      setE2eeSetupComplete(true);
      return;
    }
    let cancelled = false;
    keyProvider.setKey(passphrase).then(() => {
      if (cancelled) return;
      room.setE2EEEnabled(true).then(() => {
        if (!cancelled) setE2eeSetupComplete(true);
      });
    });
    return () => {
      cancelled = true;
    };
  }, [e2eeEnabled, passphrase, keyProvider, room]);

  useEffect(() => {
    const disconnectOnHide = () => {
      if (!connectedRef.current) return;
      skipDisconnectRedirectRef.current = true;
      room.disconnect();
    };
    window.addEventListener('pagehide', disconnectOnHide);
    return () => window.removeEventListener('pagehide', disconnectOnHide);
  }, [room]);

  useEffect(() => {
    const onLeave = () => {
      // Unmount/refresh/tab close calls room.disconnect() — stay on /custom.
      if (skipDisconnectRedirectRef.current) {
        skipDisconnectRedirectRef.current = false;
        return;
      }
      router.push('/');
    };
    room.on(RoomEvent.Disconnected, onLeave);
    return () => {
      room.off(RoomEvent.Disconnected, onLeave);
    };
  }, [room, router]);

  const connectToRoom = useMemo(() => {
    return async () => {
      setConnectError(null);
      if (connectedRef.current) {
        skipDisconnectRedirectRef.current = true;
        await room.disconnect();
        connectedRef.current = false;
      }
      const choices = props.userChoices;
      await room.connect(props.liveKitUrl, props.token, connectOptions);
      connectedRef.current = true;
      await room.localParticipant.setCameraEnabled(choices?.videoEnabled ?? false);
      await room.localParticipant.setMicrophoneEnabled(choices?.audioEnabled ?? false);
    };
  }, [
    room,
    props.liveKitUrl,
    props.token,
    connectOptions,
    props.userChoices,
  ]);

  useEffect(() => {
    if (!e2eeSetupComplete) return;

    let cancelled = false;

    (async () => {
      try {
        await connectToRoom();
      } catch (error) {
        if (cancelled) return;
        const msg = error instanceof Error ? error.message : String(error);
        console.error('TalkshowRoom connect failed', error);
        connectedRef.current = false;
        setConnectError(msg);
      }
    })();

    return () => {
      cancelled = true;
      if (connectedRef.current) {
        connectedRef.current = false;
        skipDisconnectRedirectRef.current = true;
        void room.disconnect();
      }
    };
  }, [room, e2eeSetupComplete, connectToRoom]);

  return (
    <RoomContext.Provider value={room}>
      <div className={styles.roomShell}>
        {connectError ? (
          <ConnectionRecovery
            roomName={roomName}
            message={connectError}
            onRetry={() => {
              void connectToRoom().catch((error) => {
                const msg = error instanceof Error ? error.message : String(error);
                setConnectError(msg);
              });
            }}
            onLeave={() => router.push('/')}
          />
        ) : (
          <TalkshowView viewerLocale={viewerLocale} />
        )}
        <StartMediaButton label="Click to enable playback" />
        <DebugMode logLevel={LogLevel.debug} />
      </div>
    </RoomContext.Provider>
  );
}
