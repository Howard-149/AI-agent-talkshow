import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, type VRM } from '@pixiv/three-vrm';

const bufferCache = new Map<string, Promise<ArrayBuffer>>();

export type VrmFramingOptions = {
  /** Camera distance multiplier — higher = farther / smaller face. */
  distMul?: number;
  lookDown?: number;
  lookUp?: number;
  camLift?: number;
  /** Override camera vertical FOV (degrees). */
  fov?: number;
};

const DEFAULT_FRAMING: Required<VrmFramingOptions> = {
  distMul: 2.35,
  lookDown: 0,
  lookUp: 0,
  camLift: 0.02,
  fov: 32,
};

function fetchVrmBuffer(url: string): Promise<ArrayBuffer> {
  let pending = bufferCache.get(url);
  if (!pending) {
    pending = fetch(url).then((res) => {
      if (!res.ok) {
        throw new Error(`VRM fetch failed: ${url} (${res.status})`);
      }
      return res.arrayBuffer();
    });
    bufferCache.set(url, pending);
  }
  return pending;
}

function prepareVrm(gltf: { scene: THREE.Object3D; userData: { vrm?: VRM } }): VRM {
  const vrm = gltf.userData.vrm;
  if (!vrm) {
    throw new Error('GLTF is not a VRM file');
  }
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  VRMUtils.combineMorphs(vrm);
  vrm.scene.traverse((obj) => {
    obj.frustumCulled = false;
  });
  return vrm;
}

export async function loadVrmFromUrl(url: string): Promise<VRM> {
  const buffer = await fetchVrmBuffer(url);
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));

  const gltf = await new Promise<{
    scene: THREE.Object3D;
    userData: { vrm?: VRM };
  }>((resolve, reject) => {
    loader.parse(
      buffer,
      url,
      (parsed) => resolve(parsed),
      (err) => reject(err instanceof Error ? err : new Error(String(err))),
    );
  });

  return prepareVrm(gltf);
}

function worldPos(node: THREE.Object3D, out: THREE.Vector3): THREE.Vector3 {
  node.updateWorldMatrix(true, false);
  return out.setFromMatrixPosition(node.matrixWorld);
}

function modelFrontDirection(vrm: VRM, out: THREE.Vector3): THREE.Vector3 {
  vrm.scene.updateMatrixWorld(true);
  return out.set(0, 0, 1).transformDirection(vrm.scene.matrixWorld).normalize();
}

/** Measured ~0.90 of body height for Seed-san (no eye bones). */
const SEED_SAN_EYE_Y_RATIO = 0.898;

function faceFromBodyRatio(vrm: VRM): {
  lookAt: THREE.Vector3;
  headSize: number;
  camForward: THREE.Vector3;
} {
  vrm.scene.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(vrm.scene);
  const size = box.getSize(new THREE.Vector3());
  const eyeY = box.min.y + size.y * SEED_SAN_EYE_Y_RATIO;

  const head = vrm.humanoid?.getRawBoneNode('head');
  const lookAt = new THREE.Vector3();
  if (head) {
    worldPos(head, lookAt);
    lookAt.y = eyeY;
  } else {
    lookAt.set((box.min.x + box.max.x) * 0.5, eyeY, (box.min.z + box.max.z) * 0.5);
  }

  const sliceMinY = box.min.y + size.y * 0.83;
  const sliceH = Math.max(box.max.y - sliceMinY, size.y * 0.12);

  return {
    lookAt,
    headSize: Math.max(sliceH * 1.05, size.y * 0.21),
    camForward: modelFrontDirection(vrm, new THREE.Vector3()),
  };
}

function faceFromHumanoid(vrm: VRM): {
  lookAt: THREE.Vector3;
  headSize: number;
  camForward: THREE.Vector3;
} | null {
  const humanoid = vrm.humanoid;
  if (!humanoid) return null;

  humanoid.update();
  vrm.scene.updateMatrixWorld(true);

  const leftEye = humanoid.getRawBoneNode('leftEye');
  const rightEye = humanoid.getRawBoneNode('rightEye');
  const head = humanoid.getRawBoneNode('head');
  const neck =
    humanoid.getRawBoneNode('neck') ?? humanoid.getRawBoneNode('upperChest');

  if (!leftEye || !rightEye) {
    return faceFromBodyRatio(vrm);
  }

  const lookAt = new THREE.Vector3();
  const tmpL = new THREE.Vector3();
  const tmpR = new THREE.Vector3();
  const tmpHead = new THREE.Vector3();
  const tmpNeck = new THREE.Vector3();
  const modelFront = modelFrontDirection(vrm, new THREE.Vector3());

  worldPos(leftEye, tmpL);
  worldPos(rightEye, tmpR);
  lookAt.copy(tmpL).add(tmpR).multiplyScalar(0.5);

  let headSize = Math.max(tmpL.distanceTo(tmpR) * 2.85, 0.16);
  if (head && neck) {
    worldPos(head, tmpHead);
    worldPos(neck, tmpNeck);
    headSize = Math.max(headSize, tmpHead.distanceTo(tmpNeck) * 2.5);
  }

  let camForward: THREE.Vector3;
  if (head && neck) {
    worldPos(head, tmpHead);
    worldPos(neck, tmpNeck);
    const eyeDir = tmpR.clone().sub(tmpL).normalize();
    const spineUp = tmpHead.clone().sub(tmpNeck).normalize();
    camForward = new THREE.Vector3().crossVectors(eyeDir, spineUp).normalize();
    if (camForward.dot(modelFront) < 0) {
      camForward.negate();
    }
  } else {
    camForward = modelFront.clone();
  }

  return { lookAt, headSize, camForward };
}

/** Frame camera on face — full-body VRM samples need ratio/bone crop, not whole-scene bbox. */
export function frameHeadBust(
  vrm: VRM,
  camera: THREE.PerspectiveCamera,
  framing?: VrmFramingOptions,
): void {
  const opts: Required<VrmFramingOptions> = { ...DEFAULT_FRAMING, ...framing };
  camera.fov = opts.fov;
  camera.updateProjectionMatrix();

  const face = faceFromHumanoid(vrm);
  if (!face) {
    return;
  }

  const lookAt = face.lookAt.clone();
  lookAt.y += face.headSize * opts.lookUp;
  lookAt.y -= face.headSize * opts.lookDown;

  const fovRad = (camera.fov * Math.PI) / 180;
  const dist = (face.headSize * opts.distMul) / (2 * Math.tan(fovRad / 2));

  const camPos = lookAt.clone().add(face.camForward.clone().multiplyScalar(dist));
  camPos.y += face.headSize * opts.camLift;

  camera.position.copy(camPos);
  camera.lookAt(lookAt);
  camera.updateProjectionMatrix();
}

export function disposeVrm(vrm: VRM, renderer: THREE.WebGLRenderer): void {
  VRMUtils.deepDispose(vrm.scene);
  renderer.dispose();
}
