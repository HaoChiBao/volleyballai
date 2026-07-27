"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo, useState } from "react";
import {
  BoxGeometry,
  EdgesGeometry,
  PerspectiveCamera,
  Vector3,
} from "three";
import { poseToThreeCamera, type CameraPose } from "@volleyballai/court-math";
import type {
  BallTracksFile,
  Calibration,
  CalibrationCamera,
  PlayersTracksFile,
} from "@volleyballai/types";
import { bracketFrames, lerp } from "@/lib/trackInterp";

export type Court3dSample = {
  t: number;
  players: { track_id: number; x: number; y: number; z: number }[];
  ball?: { x: number; y: number; z: number } | null;
};

export type Court3dFile = {
  court: { length_m: number; width_m: number };
  samples: Court3dSample[];
  camera?: CalibrationCamera | null;
};

const COLORS = ["#111111", "#333333", "#555555", "#777777", "#999999", "#bbbbbb"];

function toPose(cam: CalibrationCamera | CameraPose): CameraPose {
  return {
    position: cam.position,
    R: cam.R,
    t: cam.t,
    fx: cam.fx,
    fy: cam.fy,
    cx: cam.cx,
    cy: cam.cy,
    image_width: cam.image_width,
    image_height: cam.image_height,
    fov_y_deg: cam.fov_y_deg,
  };
}

function CourtMesh({ length, width }: { length: number; width: number }) {
  const edges = useMemo(
    () => new EdgesGeometry(new BoxGeometry(length, 0.01, width)),
    [length, width],
  );

  return (
    <group>
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[length / 2, 0, width / 2]}
        receiveShadow
      >
        <planeGeometry args={[length, width]} />
        <meshStandardMaterial color="#d0d0d0" />
      </mesh>
      {/* sideline / endline marks */}
      <mesh position={[length / 2, 0.01, 0]}>
        <boxGeometry args={[length, 0.02, 0.05]} />
        <meshStandardMaterial color="#0a0a0a" />
      </mesh>
      <mesh position={[length / 2, 0.01, width]}>
        <boxGeometry args={[length, 0.02, 0.05]} />
        <meshStandardMaterial color="#0a0a0a" />
      </mesh>
      <mesh position={[0, 0.01, width / 2]}>
        <boxGeometry args={[0.05, 0.02, width]} />
        <meshStandardMaterial color="#0a0a0a" />
      </mesh>
      <mesh position={[length, 0.01, width / 2]}>
        <boxGeometry args={[0.05, 0.02, width]} />
        <meshStandardMaterial color="#0a0a0a" />
      </mesh>
      {/* center line + net */}
      <mesh position={[length / 2, 0.02, width / 2]}>
        <boxGeometry args={[0.06, 0.02, width]} />
        <meshStandardMaterial color="#0a0a0a" />
      </mesh>
      <mesh position={[length / 2, 1.2, width / 2]}>
        <boxGeometry args={[0.04, 2.4, width]} />
        <meshStandardMaterial color="#1a1a1a" transparent opacity={0.45} />
      </mesh>
      <lineSegments geometry={edges} position={[length / 2, 0, width / 2]}>
        <lineBasicMaterial color="#0a0a0a" />
      </lineSegments>
    </group>
  );
}

function Players({
  markers,
}: {
  markers: { track_id: number; x: number; y: number; z: number }[];
}) {
  return (
    <group>
      {markers.map((m, i) => (
        <mesh key={m.track_id} position={[m.x, 0.9, m.y]} castShadow>
          <capsuleGeometry args={[0.25, 1.2, 4, 8]} />
          <meshStandardMaterial color={COLORS[i % COLORS.length]} />
        </mesh>
      ))}
    </group>
  );
}

function Ball({ ball }: { ball: { x: number; y: number; z: number } }) {
  return (
    <mesh position={[ball.x, ball.z, ball.y]} castShadow>
      <sphereGeometry args={[0.105, 24, 24]} />
      <meshStandardMaterial color="#0a0a0a" />
    </mesh>
  );
}

/** Lock Three camera to recovered video view each frame. */
function MatchedCamera({
  pose,
  enabled,
}: {
  pose: CameraPose;
  enabled: boolean;
}) {
  const { camera } = useThree();
  const three = useMemo(() => poseToThreeCamera(pose), [pose]);
  const look = useMemo(() => new Vector3(...three.lookAt), [three]);

  useFrame(() => {
    if (!enabled) return;
    camera.position.set(...three.position);
    camera.up.set(0, 1, 0);
    camera.lookAt(look);
    if (camera instanceof PerspectiveCamera) {
      camera.fov = three.fov;
      camera.updateProjectionMatrix();
    }
  });

  return null;
}

function interpolatePlayers(
  tracks: PlayersTracksFile | null,
  t: number,
): { track_id: number; x: number; y: number; z: number }[] {
  if (!tracks) return [];
  const out: { track_id: number; x: number; y: number; z: number }[] = [];
  for (const p of tracks.players) {
    const withCourt = p.frames.filter((f) => f.court_xy != null);
    if (!withCourt.length) continue;
    const br = bracketFrames(withCourt, t, 0.45);
    if (!br) continue;
    if (br.kind === "lerp") {
      const a = br.a.court_xy!;
      const b = br.b.court_xy!;
      out.push({
        track_id: p.track_id,
        x: lerp(a[0], b[0], br.u),
        y: lerp(a[1], b[1], br.u),
        z: 0,
      });
    } else {
      const xy = br.frame.court_xy!;
      out.push({ track_id: p.track_id, x: xy[0], y: xy[1], z: 0 });
    }
  }
  return out;
}

function interpolateBall3d(
  ball: BallTracksFile | null,
  t: number,
): { x: number; y: number; z: number } | null {
  if (!ball?.frames.length) return null;
  const frames = ball.frames.filter((f) => f.court_xyz != null);
  if (!frames.length) return null;
  const br = bracketFrames(frames, t, 0.2);
  if (!br) return null;
  if (br.kind === "lerp") {
    const a = br.a.court_xyz!;
    const b = br.b.court_xyz!;
    return {
      x: lerp(a[0], b[0], br.u),
      y: lerp(a[1], b[1], br.u),
      z: lerp(a[2], b[2], br.u),
    };
  }
  const xyz = br.frame.court_xyz!;
  return { x: xyz[0], y: xyz[1], z: xyz[2] };
}

export function Court3D({
  court3d,
  calibration,
  tracks,
  ball,
  currentTime,
  compact,
}: {
  court3d: Court3dFile | null;
  calibration?: Calibration | null;
  tracks?: PlayersTracksFile | null;
  ball?: BallTracksFile | null;
  currentTime: number;
  /** Fill parent height (side-by-side pane) */
  compact?: boolean;
}) {
  const cameraPose = useMemo(() => {
    const cam = calibration?.camera ?? court3d?.camera ?? null;
    return cam ? toPose(cam) : null;
  }, [calibration?.camera, court3d?.camera]);

  // Orbit free by default so the side pane is always draggable; matched cam is opt-in.
  const [matchView, setMatchView] = useState(false);

  const livePlayers = useMemo(
    () => interpolatePlayers(tracks ?? null, currentTime),
    [tracks, currentTime],
  );
  const liveBall = useMemo(
    () => interpolateBall3d(ball ?? null, currentTime),
    [ball, currentTime],
  );

  const sample = useMemo(() => {
    if (!court3d?.samples?.length) return null;
    return court3d.samples.reduce((a, b) =>
      Math.abs(a.t - currentTime) < Math.abs(b.t - currentTime) ? a : b,
    );
  }, [court3d, currentTime]);

  const markers =
    livePlayers.length > 0 ? livePlayers : (sample?.players ?? []);
  const ballPos = liveBall ?? sample?.ball ?? null;

  const length = court3d?.court.length_m ?? calibration?.court.length_m ?? 18;
  const width = court3d?.court.width_m ?? calibration?.court.width_m ?? 9;

  const threeCam = useMemo(
    () => (cameraPose ? poseToThreeCamera(cameraPose) : null),
    [cameraPose],
  );

  const defaultPos: [number, number, number] = threeCam?.position ?? [
    9, 14, 22,
  ];
  const defaultFov = threeCam?.fov ?? 42;
  const lookTarget = threeCam?.lookAt ?? [length / 2, 0, width / 2];

  return (
    <div className={compact ? "court3d-pane stack" : "card stack"}>
      <div className="row between">
        <h2>{compact ? "3D court" : "3D court"}</h2>
        <div className="row" style={{ gap: "0.5rem" }}>
          {cameraPose ? (
            <button
              type="button"
              className={`toggle-chip${matchView ? " active" : ""}`}
              onClick={() => setMatchView((v) => !v)}
              title={
                matchView
                  ? "Switch to free orbit (drag to look around)"
                  : "Lock camera to the video viewpoint"
              }
            >
              {matchView ? "Matched cam" : "Orbit free"}
            </button>
          ) : null}
          <span className="meta-line">
            {markers.length} players
            {ballPos ? " + ball" : ""} @ {currentTime.toFixed(2)}s
            {!matchView ? " · drag to orbit" : ""}
          </span>
        </div>
      </div>
      <div
        className="video-shell court3d-canvas"
        style={
          compact
            ? {
                flex: 1,
                minHeight: 280,
                background: "#e8e8e8",
                touchAction: "none",
                cursor: matchView ? "default" : "grab",
              }
            : {
                height: 360,
                background: "#e8e8e8",
                touchAction: "none",
                cursor: matchView ? "default" : "grab",
              }
        }
      >
        <Canvas
          shadows
          camera={{ position: defaultPos, fov: defaultFov }}
          style={{ width: "100%", height: "100%" }}
          // Keep R3F from swallowing scroll on the page; orbit uses pointer drag.
          onPointerDown={(e) => {
            (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
          }}
        >
          <ambientLight intensity={0.75} />
          <directionalLight position={[12, 20, 8]} intensity={1.1} castShadow />
          <CourtMesh length={length} width={width} />
          <Players markers={markers} />
          {ballPos ? <Ball ball={ballPos} /> : null}
          {cameraPose ? (
            <MatchedCamera pose={cameraPose} enabled={matchView} />
          ) : null}
          <OrbitControls
            makeDefault
            enabled={!matchView || !cameraPose}
            enableRotate
            enablePan
            enableZoom
            target={lookTarget as [number, number, number]}
            enableDamping
            dampingFactor={0.08}
            minDistance={3}
            maxDistance={60}
            maxPolarAngle={Math.PI * 0.49}
          />
        </Canvas>
      </div>
      {!compact ? (
        <p className="hint">
          {cameraPose
            ? matchView
              ? "Matched camera locks to the video viewpoint. Click Orbit free to drag/pan/zoom."
              : "Drag to orbit, right-drag or two-finger to pan, scroll to zoom. Toggle Matched cam to lock to the video view."
            : "Calibrate the court (YOLO keypoints or manual lines) to recover the video camera and place the ball in 3D."}
        </p>
      ) : null}
    </div>
  );
}
