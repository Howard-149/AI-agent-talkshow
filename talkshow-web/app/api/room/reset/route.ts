import { RoomServiceClient } from 'livekit-server-sdk';
import { NextRequest, NextResponse } from 'next/server';

/**
 * Dev recovery: delete a stuck LiveKit room (ghost participant after tab kill).
 * CAUTION: unauthenticated — same as Meet record routes; do not expose on public deploy.
 */
export async function POST(req: NextRequest) {
  try {
    const roomName = req.nextUrl.searchParams.get('roomName');
    if (!roomName) {
      return new NextResponse('Missing roomName parameter', { status: 400 });
    }

    const { LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL } = process.env;
    if (!LIVEKIT_API_KEY || !LIVEKIT_API_SECRET || !LIVEKIT_URL) {
      return new NextResponse('LiveKit env not configured', { status: 500 });
    }

    const hostURL = new URL(LIVEKIT_URL);
    hostURL.protocol = 'https:';

    const rooms = new RoomServiceClient(hostURL.origin, LIVEKIT_API_KEY, LIVEKIT_API_SECRET);
    try {
      await rooms.deleteRoom(roomName);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (message.toLowerCase().includes('not found')) {
        return NextResponse.json({ ok: true, roomName, note: 'room already gone' });
      }
      throw err;
    }

    return NextResponse.json({ ok: true, roomName });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'room reset failed';
    return new NextResponse(message, { status: 500 });
  }
}
