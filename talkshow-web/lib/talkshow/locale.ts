/** Viewer preferred locale for transcript / media (en | zh). */

export type TalkshowLocale = 'en' | 'zh';

const STORAGE_KEY = 'talkshow.viewerLocale';

export function normalizeTalkshowLocale(raw: string | null | undefined): TalkshowLocale {
  const code = (raw || '').trim().toLowerCase();
  if (code.startsWith('zh')) return 'zh';
  return 'en';
}

export function readStoredViewerLocale(): TalkshowLocale {
  if (typeof window === 'undefined') return 'en';
  try {
    return normalizeTalkshowLocale(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return 'en';
  }
}

export function writeStoredViewerLocale(locale: TalkshowLocale): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, locale);
  } catch {
    /* ignore quota / private mode */
  }
}

/** Locale baked into the LiveKit access token metadata at join (agent reads this). */
export function localeFromAccessToken(token: string): TalkshowLocale | null {
  const parts = token.split('.');
  if (parts.length < 2) return null;
  try {
    const b64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const padded = b64 + '='.repeat((4 - (b64.length % 4)) % 4);
    const payload = JSON.parse(atob(padded)) as { metadata?: string };
    const raw = payload.metadata;
    if (!raw || typeof raw !== 'string') return null;
    const meta = JSON.parse(raw) as { locale?: string };
    if (meta?.locale == null || meta.locale === '') return null;
    return normalizeTalkshowLocale(String(meta.locale));
  } catch {
    return null;
  }
}

export function pickLocalizedText(
  text: string,
  texts: Partial<Record<TalkshowLocale, string>> | undefined,
  locale: TalkshowLocale,
): string {
  const localized = texts?.[locale]?.trim();
  if (localized) return localized;
  const en = texts?.en?.trim();
  if (en) return en;
  return text;
}

export function avatarTrackNameForLocale(locale: TalkshowLocale): string {
  return locale === 'en' ? 'talkshow-avatar' : `talkshow-avatar-${locale}`;
}

export function audioTrackNameForLocale(locale: TalkshowLocale): string {
  return `talkshow-audio-${locale}`;
}
