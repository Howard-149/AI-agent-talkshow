import { describe, expect, it } from 'vitest';

import { isGhostViewer } from './roles';

describe('isGhostViewer', () => {
  it('matches the ghost-viewer identity prefix', () => {
    expect(isGhostViewer('ghost-viewer')).toBe(true);
    expect(isGhostViewer('ghost-viewer-1')).toBe(true);
  });

  it('matches token metadata', () => {
    expect(isGhostViewer('cameraman', '{"ghostViewer":true}')).toBe(true);
  });

  it('ignores the injecting ghost guest', () => {
    expect(isGhostViewer('ghost-guest', '{"ghost":true}')).toBe(false);
    expect(isGhostViewer('howard')).toBe(false);
  });
});
