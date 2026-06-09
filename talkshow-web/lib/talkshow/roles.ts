export type PanelRole = string;

export type PanelistDef = {
  role: PanelRole;
  name: string;
  label: string;
  color: string;
};

export const UI_TOPIC = 'talkshow/ui';

export type UiEvent =
  | { type: 'panel_roster'; scenario: string; members: PanelistDef[] }
  | { type: 'role_active'; role: PanelRole; name: string }
  | { type: 'role_idle' }
  | { type: 'floor_pending'; active: boolean }
  | {
      type: 'transcript';
      role: string;
      speaker: string;
      text: string;
      final: boolean;
      step?: string;
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
