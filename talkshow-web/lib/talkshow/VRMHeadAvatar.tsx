'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';

import { disposeVrm, frameHeadBust, loadVrmFromUrl, type VrmFramingOptions } from '@/lib/talkshow/vrm/loadVrm';
import type { VRM } from '@pixiv/three-vrm';

import styles from '@/styles/TalkshowStage.module.css';

type Props = {
  vrmUrl: string;
  isSpeaking: boolean;
  scale?: number;
  framing?: VrmFramingOptions;
  onLoadFailed?: () => void;
};

const IDLE_FRAME_MS = 800;
const SPEAKING_FRAME_MS = 1000 / 30;

export function VRMHeadAvatar({ vrmUrl, isSpeaking, scale = 1, framing, onLoadFailed }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const speakingRef = useRef(isSpeaking);

  useEffect(() => {
    speakingRef.current = isSpeaking;
  }, [isSpeaking]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const width = host.clientWidth || 128;
    const height = host.clientHeight || 128;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setClearColor(0x000000, 0);
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(framing?.fov ?? 32, width / height, 0.1, 20);

    const keyLight = new THREE.DirectionalLight(0xffffff, Math.PI * 0.9);
    keyLight.position.set(0.8, 1.2, 1.0).normalize();
    scene.add(keyLight);
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));

    let vrm: VRM | null = null;
    let raf = 0;
    let cancelled = false;
    let lastFrameMs = 0;
    const clock = new THREE.Clock();

    const renderFrame = (t: number) => {
      if (!vrm) return;
      const dt = clock.getDelta();
      const em = vrm.expressionManager;
      if (em) {
        if (speakingRef.current) {
          em.setValue('aa', 0.25 + 0.45 * Math.abs(Math.sin(t * 11)));
        } else {
          em.setValue('aa', 0);
        }
        em.setValue('blink', Math.sin(t * 2.8) > 0.97 ? 1 : 0);
        em.update();
      }
      // No lookAt — tracking the camera twists head/neck when framing is bust-only.
      vrm.update(dt);
      renderer.render(scene, camera);
    };

    const animate = (now: number) => {
      raf = requestAnimationFrame(animate);
      if (!vrm) return;
      const speaking = speakingRef.current;
      const minGap = speaking ? SPEAKING_FRAME_MS : IDLE_FRAME_MS;
      if (now - lastFrameMs < minGap) return;
      lastFrameMs = now;
      renderFrame(clock.elapsedTime);
    };

    loadVrmFromUrl(vrmUrl)
      .then((loaded) => {
        if (cancelled) {
          disposeVrm(loaded, renderer);
          return;
        }
        vrm = loaded;
        if (scale !== 1) {
          loaded.scene.scale.setScalar(scale);
        }
        scene.add(loaded.scene);
        frameHeadBust(loaded, camera, framing);
        renderFrame(0);
        animate(performance.now());
      })
      .catch((err) => {
        console.error('VRMHeadAvatar load failed', vrmUrl, err);
        onLoadFailed?.();
        renderer.dispose();
        if (renderer.domElement.parentElement === host) {
          host.removeChild(renderer.domElement);
        }
      });

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      if (vrm) {
        disposeVrm(vrm, renderer);
        vrm = null;
      } else {
        renderer.dispose();
      }
      if (renderer.domElement.parentElement === host) {
        host.removeChild(renderer.domElement);
      }
    };
  }, [vrmUrl, scale, framing?.distMul, framing?.lookDown, framing?.lookUp, framing?.camLift, framing?.fov]);

  return (
    <div
      ref={hostRef}
      className={styles.avatarCanvas}
      aria-hidden
      data-speaking={isSpeaking ? 'true' : 'false'}
    />
  );
}
