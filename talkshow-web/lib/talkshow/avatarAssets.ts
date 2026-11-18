/** Static DyStream assets: /avatars/* → public symlinks → avatar/assets (single source). */

/** Which avatar engine renders the faces (the agent's TALKSHOW_AVATAR_ENGINE). */
export const AVATAR_ENGINE: 'dystream' | 'avtr1' =
  process.env.NEXT_PUBLIC_AVATAR_ENGINE === 'avtr1' ? 'avtr1' : 'dystream';

// Idle loops must share the speaking frames' framing: each engine has its own set
// (AVTR-1's from deploy/avtr1/render_idle_loops.py).
const IDLE_SUFFIX = AVATAR_ENGINE === 'avtr1' ? '-idle-avtr1.mp4' : '-idle.mp4';
const LOCAL_IDLE_BY_ROLE: Record<string, string> = {
  host: `/avatars/loops/lessac${IDLE_SUFFIX}`,
  commentator: `/avatars/loops/ryan${IDLE_SUFFIX}`,
  guest: `/avatars/loops/amy${IDLE_SUFFIX}`,
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
