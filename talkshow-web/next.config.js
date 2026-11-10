const path = require('path');
const { loadEnvConfig } = require('@next/env');

// Single source of truth: repo root .env (same as agent + api/tokens.py)
// forceReload: Next has already loaded (and cached) this app dir's env by now, so a
// plain call would return that cache and never read the root .env.
loadEnvConfig(path.join(__dirname, '..'), process.env.NODE_ENV !== 'production', console, true);

/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['three', '@pixiv/three-vrm'],
  env: {
    // Browser cannot read LIVEKIT_URL; reuse root value for connect form default
    NEXT_PUBLIC_LIVEKIT_URL:
      process.env.NEXT_PUBLIC_LIVEKIT_URL || process.env.LIVEKIT_URL || '',
    NEXT_PUBLIC_DYSTREAM_ENABLED: process.env.NEXT_PUBLIC_DYSTREAM_ENABLED ?? '1',
    NEXT_PUBLIC_VRM_ENABLED: process.env.NEXT_PUBLIC_VRM_ENABLED ?? '0',
  },
  reactStrictMode: false,
  productionBrowserSourceMaps: true,
  images: {
    formats: ['image/webp'],
  },
  webpack: (config, { buildId, dev, isServer, defaultLoaders, nextRuntime, webpack }) => {
    // Important: return the modified config
    config.module.rules.push({
      test: /\.mjs$/,
      enforce: 'pre',
      use: ['source-map-loader'],
    });

    return config;
  },
  headers: async () => {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'credentialless',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
