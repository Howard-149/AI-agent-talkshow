import { randomString } from '@/lib/client-utils';
import { getLiveKitURL } from '@/lib/getLiveKitURL';
import { ConnectionDetails } from '@/lib/types';
import {
  AccessToken,
  AccessTokenOptions,
  RoomAgentDispatch,
  RoomConfiguration,
  VideoGrant,
} from 'livekit-server-sdk';
import { NextRequest, NextResponse } from 'next/server';

const API_KEY = process.env.LIVEKIT_API_KEY;
const API_SECRET = process.env.LIVEKIT_API_SECRET;
const LIVEKIT_URL = process.env.LIVEKIT_URL;
// Must match the worker's TALKSHOW_AGENT_NAME (same repo-root .env), or the room gets no agent.
const AGENT_NAME = (process.env.TALKSHOW_AGENT_NAME ?? '').trim();

const COOKIE_KEY = 'random-participant-postfix';

export async function GET(request: NextRequest) {
  try {
    // Parse query parameters
    const roomName = request.nextUrl.searchParams.get('roomName');
    const participantName = request.nextUrl.searchParams.get('participantName');
    const metadataParam = request.nextUrl.searchParams.get('metadata') ?? '';
    const localeParam = request.nextUrl.searchParams.get('locale');
    const region = request.nextUrl.searchParams.get('region');
    const aiOnly = request.nextUrl.searchParams.get('aiOnly') === '1';
    if (!LIVEKIT_URL) {
      throw new Error('LIVEKIT_URL is not defined');
    }
    const livekitServerUrl = region ? getLiveKitURL(LIVEKIT_URL, region) : LIVEKIT_URL;
    let randomParticipantPostfix = request.cookies.get(COOKIE_KEY)?.value;
    if (livekitServerUrl === undefined) {
      throw new Error('Invalid region');
    }

    if (typeof roomName !== 'string') {
      return new NextResponse('Missing required query parameter: roomName', { status: 400 });
    }
    if (participantName === null) {
      return new NextResponse('Missing required query parameter: participantName', { status: 400 });
    }

    // Locale must be in join-token metadata so the agent can localize opening TTS
    // immediately (no set_locale round-trip).
    let metadata = metadataParam;
    if (localeParam) {
      const locale = localeParam.trim().toLowerCase().startsWith('zh') ? 'zh' : 'en';
      try {
        const parsed = metadata ? JSON.parse(metadata) : {};
        const obj =
          parsed && typeof parsed === 'object' && !Array.isArray(parsed)
            ? { ...(parsed as Record<string, unknown>) }
            : {};
        obj.locale = locale;
        metadata = JSON.stringify(obj);
      } catch {
        metadata = JSON.stringify({ locale });
      }
    }

    // Generate participant token
    if (!randomParticipantPostfix) {
      randomParticipantPostfix = randomString(4);
    }
    // Laptop text mode reuses the repo's guarded ghost-turn path. Audio-only
    // Gemma deployments keep their normal participant metadata unchanged.
    if (process.env.TALKSHOW_TEXT_INPUT === '1') {
      let parsed: Record<string, unknown> = {};
      try {
        const value = metadata ? JSON.parse(metadata) : {};
        if (value && typeof value === 'object' && !Array.isArray(value)) parsed = value;
      } catch {
        // A malformed optional metadata parameter must not break local joins.
      }
      metadata = JSON.stringify({
        ...parsed,
        ghost: true,
        ...(aiOnly ? { aiOnly: true, ghostViewer: true } : {}),
      });
    }

    const participantToken = await createParticipantToken(
      {
        identity: `${participantName}__${randomParticipantPostfix}`,
        name: participantName,
        metadata,
      },
      roomName,
      aiOnly,
    );

    // Return connection details
    const data: ConnectionDetails = {
      serverUrl: livekitServerUrl,
      roomName: roomName,
      participantToken: participantToken,
      participantName: participantName,
    };
    return new NextResponse(JSON.stringify(data), {
      headers: {
        'Content-Type': 'application/json',
        'Set-Cookie': `${COOKIE_KEY}=${randomParticipantPostfix}; Path=/; HttpOnly; SameSite=Strict; Secure; Expires=${getCookieExpirationTime()}`,
      },
    });
  } catch (error) {
    if (error instanceof Error) {
      return new NextResponse(error.message, { status: 500 });
    }
  }
}

function createParticipantToken(userInfo: AccessTokenOptions, roomName: string, aiOnly = false) {
  const at = new AccessToken(API_KEY, API_SECRET, userInfo);
  at.ttl = '5m';
  const grant: VideoGrant = {
    room: roomName,
    roomJoin: true,
    canPublish: !aiOnly,
    canPublishData: true,
    canSubscribe: true,
  };
  at.addGrant(grant);
  if (AGENT_NAME) {
    at.roomConfig = new RoomConfiguration({
      agents: [new RoomAgentDispatch({ agentName: AGENT_NAME })],
    });
  }
  return at.toJwt();
}

function getCookieExpirationTime(): string {
  var now = new Date();
  var time = now.getTime();
  var expireTime = time + 60 * 120 * 1000;
  now.setTime(expireTime);
  return now.toUTCString();
}
