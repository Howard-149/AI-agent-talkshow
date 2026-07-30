/** Static DyStream assets: /avatars/* → public symlinks → avatar/assets (single source). */

const LOCAL_IDLE_BY_ROLE: Record<string, string> = {
  host: '/avatars/loops/lessac-idle.mp4',
  commentator: '/avatars/loops/ryan-idle.mp4',
  guest: '/avatars/loops/amy-idle.mp4',
};

const LOCAL_PORTRAIT_BY_ROLE: Record<string, string> = {
  host: '/avatars/portraits/lessac.png',
  commentator: '/avatars/portraits/ryan.png',
  guest: '/avatars/portraits/amy.png',
};

/** Prefer local `/avatars/*` (symlinked to `avatar/assets`) over Babel HTTP URLs. */
export function localIdleUrl(role: string): string | undefined {
  return LOCAL_IDLE_BY_ROLE[role];
}

export function localPortraitUrl(role: string): string | undefined {
  return LOCAL_PORTRAIT_BY_ROLE[role];
}
