'use client';

import React from 'react';
import { decodePassphrase } from '@/lib/client-utils';
import { TalkshowRoom } from '@/lib/talkshow/TalkshowRoom';
import { ConnectionDetails } from '@/lib/types';
import { LocalUserChoices, PreJoin } from '@livekit/components-react';
import { VideoCodec } from 'livekit-client';
import { useRouter } from 'next/navigation';
import { useSetupE2EE } from '@/lib/useSetupE2EE';

const CONN_DETAILS_ENDPOINT =
  process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT ?? '/api/connection-details';

export function PageClientImpl(props: {
  roomName: string;
  region?: string;
  hq: boolean;
  codec: VideoCodec;
  singlePeerConnection: boolean;
  aiOnly?: boolean;
}) {
  const [preJoinChoices, setPreJoinChoices] = React.useState<LocalUserChoices | undefined>(
    undefined,
  );
  const preJoinDefaults = React.useMemo(() => {
    return {
      username: '',
      videoEnabled: false,
      audioEnabled: false,
    };
  }, []);
  const [connectionDetails, setConnectionDetails] = React.useState<ConnectionDetails | undefined>(
    undefined,
  );

  const handlePreJoinSubmit = React.useCallback(
    async (values: LocalUserChoices) => {
      setPreJoinChoices(values);
      const url = new URL(CONN_DETAILS_ENDPOINT, window.location.origin);
      url.searchParams.append('roomName', props.roomName);
      url.searchParams.append('participantName', values.username);
      if (props.aiOnly) url.searchParams.append('aiOnly', '1');
      // Bake locale into join token so agent opening TTS matches without set_locale latency.
      try {
        const stored = window.localStorage.getItem('talkshow.viewerLocale');
        url.searchParams.append('locale', stored === 'zh' ? 'zh' : 'en');
      } catch {
        url.searchParams.append('locale', 'en');
      }
      if (props.region) {
        url.searchParams.append('region', props.region);
      }
      const connectionDetailsResp = await fetch(url.toString());
      const connectionDetailsData = await connectionDetailsResp.json();
      setConnectionDetails(connectionDetailsData);
    },
    [props.roomName, props.region, props.aiOnly],
  );
  const handlePreJoinError = React.useCallback((e: unknown) => console.error(e), []);

  return (
    <main data-lk-theme="default" style={{ height: '100%' }}>
      {connectionDetails === undefined || preJoinChoices === undefined ? (
        <div style={{ display: 'grid', placeItems: 'center', height: '100%' }}>
          {props.aiOnly ? (
            <div style={{ textAlign: 'center' }}>
              <h1>AI Talk Show</h1>
              <p>Lessac · Ryan · Amy</p>
              <button
                className="lk-button"
                onClick={() =>
                  void handlePreJoinSubmit({
                    username: 'Audience',
                    audioEnabled: false,
                    videoEnabled: false,
                    audioDeviceId: 'default',
                    videoDeviceId: 'default',
                  })
                }
              >
                Enter studio
              </button>
            </div>
          ) : (
            <PreJoin
              defaults={preJoinDefaults}
              onSubmit={handlePreJoinSubmit}
              onError={handlePreJoinError}
            />
          )}
        </div>
      ) : (
        <TalkshowRoomWithE2EE
          connectionDetails={connectionDetails}
          userChoices={preJoinChoices}
          singlePeerConnection={props.singlePeerConnection}
          aiOnly={props.aiOnly}
        />
      )}
    </main>
  );
}

function TalkshowRoomWithE2EE(props: {
  connectionDetails: ConnectionDetails;
  userChoices: LocalUserChoices;
  singlePeerConnection: boolean;
  aiOnly?: boolean;
}) {
  const { e2eePassphrase } = useSetupE2EE();
  const passphrase = e2eePassphrase ? decodePassphrase(e2eePassphrase) : undefined;

  const connectChoices = React.useMemo(
    () => ({
      audioEnabled: props.userChoices.audioEnabled,
      videoEnabled: props.userChoices.videoEnabled,
      audioDeviceId: props.userChoices.audioDeviceId,
      videoDeviceId: props.userChoices.videoDeviceId,
    }),
    [props.userChoices],
  );

  return (
    <TalkshowRoom
      liveKitUrl={props.connectionDetails.serverUrl}
      token={props.connectionDetails.participantToken}
      e2eePassphrase={passphrase}
      singlePeerConnection={props.singlePeerConnection}
      userChoices={connectChoices}
      aiOnly={props.aiOnly}
    />
  );
}
