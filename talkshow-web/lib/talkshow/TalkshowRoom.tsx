'use client';

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
}) {
  const keyProvider = useMemo(() => new ExternalE2EEKeyProvider(), []);
  const { worker, e2eePassphrase } = useSetupE2EE();
  const passphrase = props.e2eePassphrase ?? e2eePassphrase;
  const e2eeEnabled = !!(passphrase && worker);

  const [e2eeSetupComplete, setE2eeSetupComplete] = useState(!e2eeEnabled);
  const connectedRef = useRef(false);
  const router = useRouter();

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
    const onLeave = () => router.push('/');
    room.on(RoomEvent.Disconnected, onLeave);
    return () => {
      room.off(RoomEvent.Disconnected, onLeave);
    };
  }, [room, router]);

  useEffect(() => {
    if (!e2eeSetupComplete) return;

    let cancelled = false;
    const choices = props.userChoices;

    (async () => {
      try {
        if (!connectedRef.current) {
          await room.connect(props.liveKitUrl, props.token, connectOptions);
          connectedRef.current = true;
        }
        if (cancelled) return;
        await room.localParticipant.setCameraEnabled(choices?.videoEnabled ?? false);
        await room.localParticipant.setMicrophoneEnabled(choices?.audioEnabled ?? false);
      } catch (error) {
        console.error('TalkshowRoom connect failed', error);
        connectedRef.current = false;
      }
    })();

    return () => {
      cancelled = true;
      if (connectedRef.current) {
        connectedRef.current = false;
        void room.disconnect();
      }
    };
  }, [
    room,
    props.liveKitUrl,
    props.token,
    connectOptions,
    e2eeSetupComplete,
    props.userChoices,
  ]);

  return (
    <RoomContext.Provider value={room}>
      <div className={styles.roomShell}>
        <TalkshowView />
        <StartMediaButton label="Click to enable playback" />
        <DebugMode logLevel={LogLevel.debug} />
      </div>
    </RoomContext.Provider>
  );
}
