'use client';

import dynamic from 'next/dynamic';
import { useState } from 'react';

import type { VrmFramingOptions } from '@/lib/talkshow/vrm/loadVrm';
import type { PanelistDef } from '@/lib/talkshow/roles';
import styles from '@/styles/TalkshowStage.module.css';

const VRMHeadAvatar = dynamic(
  () => import('@/lib/talkshow/VRMHeadAvatar').then((m) => ({ default: m.VRMHeadAvatar })),
  { ssr: false },
);

const VRM_ENABLED = process.env.NEXT_PUBLIC_VRM_ENABLED !== '0';

const DEFAULT_VRM_BY_ROLE: Record<string, string> = {
  host: '/avatars/seed-san.vrm',
  commentator: '/avatars/seed-san.vrm',
  guest: '/avatars/vrm1-twist-sample.vrm',
};

const DEFAULT_FRAMING_BY_ROLE: Record<string, VrmFramingOptions> = {
  host: { distMul: 1.38, fov: 32 },
  commentator: { distMul: 1.38, fov: 32 },
  guest: { distMul: 2.08, fov: 32 },
};

type Props = {
  panelist: PanelistDef;
  isSpeaking: boolean;
};

export function PanelAvatar({ panelist, isSpeaking }: Props) {
  const [vrmFailed, setVrmFailed] = useState(false);
  const vrmUrl = panelist.avatar?.vrm || DEFAULT_VRM_BY_ROLE[panelist.role];
  const framing: VrmFramingOptions = {
    ...DEFAULT_FRAMING_BY_ROLE[panelist.role],
    ...panelist.avatar?.framing,
  };

  if (!vrmUrl || vrmFailed || !VRM_ENABLED) {
    return (
      <div className={styles.avatar} style={{ background: panelist.color }}>
        {panelist.name[0]}
      </div>
    );
  }

  return (
    <VRMHeadAvatar
      vrmUrl={vrmUrl}
      isSpeaking={isSpeaking}
      scale={panelist.avatar?.scale ?? 1}
      framing={framing}
      onLoadFailed={() => setVrmFailed(true)}
    />
  );
}
