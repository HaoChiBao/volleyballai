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

function nearestPlayers(
  tracks: PlayersTracksFile | null,
  t: number,
): { track_id: number; x: number; y: number; z: number }[] {
  if (!tracks) return [];
  const out: { track_id: number; x: number; y: number; z: number }[] = [];
  for (const p of tracks.players) {
    if (!p.frames.length) continue;
    const frame = p.frames.reduce((a, b) =>
      Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b,
    );
    if (Math.abs(frame.t - t) > 0.35 || !frame.court_xy) continue;
    out.push({
      track_id: p.track_id,
      x: frame.court_xy[0],
      y: frame.court_xy[1],
      z: 0,
    });
  }
  return out;
}

function nearestBall(
  ball: BallTracksFile | null,
  t: number,
): { x: number; y: number; z: number } | null {
  if (!ball?.frames.length) return null;
  const frame = ball.frames.reduce((a, b) =>
    Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b,
  );
  if (Math.abs(frame.t - t) > 0.35 || !frame.court_xyz) return null;
  return {
    x: frame.court_xyz[0],
    y: frame.court_xyz[1],
    z: frame.court_xyz[2],
  };
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

  const [matchView, setMatchView] = useState(true);

  const livePlayers = useMemo(
    () => nearestPlayers(tracks ?? null, currentTime),
    [tracks, currentTime],
  );
  const liveBall = useMemo(
    () => nearestBall(ball ?? null, currentTime),
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
        <h2>{compact ? "3D (matched view)" : "3D court"}</h2>
        <div className="row" style={{ gap: "0.5rem" }}>
          {cameraPose ? (
            <button
              type="button"
              className={`toggle-chip${matchView ? " active" : ""}`}
              onClick={() => setMatchView((v) => !v)}
            >
              {matchView ? "Matched cam" : "Orbit free"}
            </button>
          ) : null}
          <span className="meta-line">
            {markers.length} players
            {ballPos ? " + ball" : ""} @ {currentTime.toFixed(2)}s
          </span>
        </div>
      </div>
      <div
        className="video-shell court3d-canvas"
        style={
          compact
            ? { flex: 1, minHeight: 280, background: "#e8e8e8" }
            : { height: 360, background: "#e8e8e8" }
        }
      >
        <Canvas
          shadows
          camera={{ position: defaultPos, fov: defaultFov }}
          style={{ width: "100%", height: "100%" }}
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
            target={lookTarget as [number, number, number]}
            enableDamping
          />
        </Canvas>
      </div>
      {!compact ? (
        <p className="hint">
          {cameraPose
            ? "Matched camera uses the same view as the video (from court calibration). Toggle Orbit free to inspect."
            : "Calibrate 4 court corners to recover the video camera and place the ball in 3D."}
        </p>
      ) : null}
    </div>
  );
}
