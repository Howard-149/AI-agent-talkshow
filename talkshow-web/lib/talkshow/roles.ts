/**
 * Panelist roster types, agent participant metadata, and `talkshow/ui` event shapes for the stage.
 */
export type PanelRole = string;

export type PanelAvatarConfig = {
  portrait?: string;
  idle_video?: string;
  video_transport?: 'livekit' | 'http';
  scale?: number;
  framing?: {
    distMul?: number;
    lookDown?: number;
    lookUp?: number;
    camLift?: number;
    fov?: number;
  };
};

export type PanelistDef = {
  role: PanelRole;
  name: string;
  label: string;
  color: string;
  avatar?: PanelAvatarConfig;
};

export const UI_TOPIC = 'talkshow/ui';

/** Agent → browser events on `talkshow/ui`. AI highlight + transcript gated on client audio. */
export type UiEvent =
  | { type: 'reactive_ready'; enabled: boolean }
  | { type: 'reactive_reset'; mode: string; topic: string }
  | { type: 'reactive_error'; message: string }
  | { type: 'reactive_generating'; role: string; name: string; step: string }
  | {
      type: 'reactive_state';
      active: boolean;
      role: string;
      name: string;
      epoch: number;
      step?: string;
      mode: string;
      heard: string;
      stop_ack_ms?: number;
      interrupted?: boolean;
    }
  | { type: 'reactive_complete'; mode: string; outcome: string }
  | {
      type: 'reactive_decision';
      role: string;
      name: string;
      status: string;
      rule: string;
      reason: string;
      source: string;
      epoch?: number;
    }
  | { type: 'panel_roster'; scenario: string; members: PanelistDef[] }
  | {
      type: 'role_active';
      role: PanelRole;
      name: string;
      /** Closed-set mood from agent [emotion] tag (avatar expression later). */
      emotion?: string;
    }
  | { type: 'role_idle' }
  | {
      type: 'avatar_clip';
      role: PanelRole;
      url?: string;
      step?: string;
      transport?: 'livekit' | 'http';
    }
  | { type: 'floor_pending'; active: boolean }
  | {
      type: 'transcript_partial';
      utterance_id: string;
      epoch: number;
      role: string;
      speaker: string;
      text: string;
      final: false;
      step?: string;
      source: 'playout_sync';
    }
  | {
      type: 'pad_state';
      role: PanelRole;
      /** Current (P, A, D) in [-1, 1]. */
      pad: [number, number, number];
      /** Spoken MSP label (what TTS uses). */
      emotion: string;
      /** Last appraisal feeling phrase, when this update came from an appraisal. */
      feeling?: string;
    }
  | {
      type: 'transcript';
      role: string;
      speaker: string;
      text: string;
      final: boolean;
      step?: string;
      texts?: { en?: string; zh?: string };
      utterance_id?: string;
      interrupted?: boolean;
    }
  | {
      type: 'hand_raise';
      role: PanelRole;
      name: string;
      raised: boolean;
      reason?: string;
      topic?: string;
    }
  | {
      type: 'floor_grant';
      role: PanelRole;
      name: string;
      reason?: string;
    }
  | {
      type: 'queue_state';
      queue: Array<{
        role: PanelRole;
        name: string;
        reason?: string;
        topic?: string;
      }>;
      phase?: string;
    };

export type AgentParticipantMetadata = {
  talkshowAgent?: boolean;
  scenario?: string;
  panelRoster?: PanelistDef[];
};

export function parseUiEvent(raw: Uint8Array): UiEvent | null {
  try {
    const text = new TextDecoder().decode(raw);
    return JSON.parse(text) as UiEvent;
  } catch {
    return null;
  }
}

export function isGhostViewer(identity: string, metadata?: string): boolean {
  if ((identity || '').startsWith('ghost-viewer')) {
    return true;
  }
  if (!metadata) {
    return false;
  }
  try {
    const meta = JSON.parse(metadata) as { ghostViewer?: boolean };
    return Boolean(meta?.ghostViewer);
  } catch {
    return false;
  }
}

export function parseAgentMetadata(raw: string | undefined): AgentParticipantMetadata | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AgentParticipantMetadata;
  } catch {
    return null;
  }
}

export function rosterFromParticipants(
  participants: Array<{ metadata?: string }>,
): PanelistDef[] | null {
  for (const p of participants) {
    const meta = parseAgentMetadata(p.metadata);
    if (meta?.talkshowAgent && meta.panelRoster?.length) {
      return meta.panelRoster;
    }
  }
  return null;
}
