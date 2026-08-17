'use client';

/**
 * Tracks HTTP speech clips and LiveKit warm-attach role so avatar video can preroll before playout.
 */
import { useCallback, useState } from 'react';

import type { PanelRole, UiEvent } from '@/lib/talkshow/roles';

export type AvatarClip = {
  url: string;
  step?: string;
};

/**
 * HTTP speech clips + LiveKit warm-attach role.
 * `avatar_clip` transport=livekit fires before preroll finishes — mount the
 * agent video element under idle so frame 0 is not lost when playout starts.
 */
export function useAvatarClips() {
  const [clipsByRole, setClipsByRole] = useState<Record<string, AvatarClip>>({});
  const [warmingRole, setWarmingRole] = useState<PanelRole | null>(null);

  const handleUiEvent = useCallback((ev: UiEvent) => {
    if (ev.type === 'avatar_clip') {
      if (ev.transport === 'livekit') {
        setWarmingRole(ev.role);
        return;
      }
      if (ev.url) {
        setClipsByRole((prev) => ({
          ...prev,
          [ev.role]: { url: ev.url, step: ev.step },
        }));
      }
      return;
    }
    if (ev.type === 'role_idle') {
      setWarmingRole(null);
    }
  }, []);

  const clipForRole = useCallback(
    (role: PanelRole) => clipsByRole[role]?.url,
    [clipsByRole],
  );

  return { clipsByRole, clipForRole, warmingRole, handleUiEvent };
}
