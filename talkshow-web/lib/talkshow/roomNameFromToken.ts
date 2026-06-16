/** Read LiveKit room name from a JWT access token (client-side, no verification). */
export function roomNameFromAccessToken(token: string): string | null {
  const parts = token.split('.');
  if (parts.length < 2) return null;
  try {
    const b64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const payload = JSON.parse(atob(b64)) as { video?: { room?: string } };
    return payload.video?.room ?? null;
  } catch {
    return null;
  }
}
