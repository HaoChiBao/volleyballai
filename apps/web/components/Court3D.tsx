"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
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
  NetTracksFile,
  PlayersTracksFile,
} from "@volleyballai/types";
import { bracketFrames, lerp } from "@/lib/trackInterp";
import { netCameraAtTime } from "@/lib/netTracks";
import {
  playerColorForTrackId,
  playerLabelForTrackId,
} from "@/lib/playerStyle";

const GaussianSplat = dynamic(
  () => import("./GaussianSplat").then((m) => m.GaussianSplat),
  { ssr: false },
);

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
  roster,
}: {
  markers: { track_id: number; x: number; y: number; z: number }[];
  /** Full `tracks.players` list — same order as 2D playlist #1, #2, … */
  roster?: { track_id: number }[];
}) {
  return (
    <group>
      {markers.map((m) => {
        const color = playerColorForTrackId(roster, m.track_id);
        const label = playerLabelForTrackId(roster, m.track_id);
        return (
          <group key={m.track_id} position={[m.x, 0, m.y]}>
            <mesh position={[0, 0.9, 0]} castShadow>
              <capsuleGeometry args={[0.25, 1.2, 4, 8]} />
              <meshStandardMaterial color={color} />
            </mesh>
            {label != null ? (
              <Html
                position={[0, 2.05, 0]}
                center
                style={{
                  pointerEvents: "none",
                  userSelect: "none",
                  fontFamily: "ui-sans-serif, system-ui, sans-serif",
                  fontWeight: 700,
                  fontSize: "12px",
                  color,
                  textShadow: "0 0 3px #fff, 0 0 2px #fff, 0 1px 2px rgba(0,0,0,0.45)",
                  whiteSpace: "nowrap",
                }}
              >
                #{label}
              </Html>
            ) : null}
          </group>
        );
      })}
    </group>
  );
}

function Ball({
  ball,
  color = "#0a0a0a",
}: {
  ball: { x: number; y: number; z: number };
  color?: string;
}) {
  return (
    <mesh position={[ball.x, ball.z, ball.y]} castShadow>
      <sphereGeometry args={[0.105, 24, 24]} />
      <meshStandardMaterial color={color} />
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
  // Match AnalysisPlayer: bridge occlusion gaps so the 3D ball does not blink out.
  const br = bracketFrames(frames, t, 1.5);
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
  ballYolo,
  ballWasb,
  netTracks,
  currentTime,
  compact,
  splatUrl,
}: {
  court3d: Court3dFile | null;
  calibration?: Calibration | null;
  tracks?: PlayersTracksFile | null;
  ball?: BallTracksFile | null;
  /** SetOptics YOLO comparison tracks (optional). */
  ballYolo?: BallTracksFile | null;
  /** WASB HRNet raw comparison tracks (optional). */
  ballWasb?: BallTracksFile | null;
  /** Settle → net PnP cameras; overrides single calibration.camera by time. */
  netTracks?: NetTracksFile | null;
  currentTime: number;
  /** Fill parent height (side-by-side pane) */
  compact?: boolean;
  /** Optional Gaussian splat .ply URL (Modal Nerfstudio export) */
  splatUrl?: string | null;
}) {
  const cameraPose = useMemo(() => {
    const fromNet = netCameraAtTime(netTracks, currentTime);
    const cam: CalibrationCamera | null =
      fromNet ?? calibration?.camera ?? court3d?.camera ?? null;
    return cam ? toPose(cam) : null;
  }, [netTracks, currentTime, calibration?.camera, court3d?.camera]);

  // Orbit free by default so drag/pan/zoom always works; matched cam is opt-in.
  const [matchView, setMatchView] = useState(false);
  const [showSplat, setShowSplat] = useState(Boolean(splatUrl));
  const [showCourtMesh, setShowCourtMesh] = useState(!splatUrl);
  const [showPlayers, setShowPlayers] = useState(true);
  const [showBallVballnet, setShowBallVballnet] = useState(true);
  const [showBallYolo, setShowBallYolo] = useState(true);
  const [showBallWasb, setShowBallWasb] = useState(true);

  useEffect(() => {
    if (splatUrl) {
      setShowSplat(true);
      setShowCourtMesh(false);
    }
  }, [splatUrl]);

  const livePlayers = useMemo(
    () => interpolatePlayers(tracks ?? null, currentTime),
    [tracks, currentTime],
  );
  const liveBall = useMemo(
    () => interpolateBall3d(ball ?? null, currentTime),
    [ball, currentTime],
  );
  const liveBallYolo = useMemo(
    () => interpolateBall3d(ballYolo ?? null, currentTime),
    [ballYolo, currentTime],
  );
  const liveBallWasb = useMemo(
    () => interpolateBall3d(ballWasb ?? null, currentTime),
    [ballWasb, currentTime],
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
  const ballYoloPos = liveBallYolo;
  const ballWasbPos = liveBallWasb;
  const hasYoloBall = Boolean(ballYolo?.frames?.length);
  const hasWasbBall = Boolean(ballWasb?.frames?.length);

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
  // Stable orbit pivot — do not retarget from matched lookAt while scrubbing
  // (that fights the user and makes controls feel broken).
  const orbitTarget = useMemo(
    (): [number, number, number] => [length / 2, 0, width / 2],
    [length, width],
  );

  return (
    <div className={compact ? "court3d-pane stack" : "card stack"}>
      <div className="row between">
        <h2>{compact ? "3D court" : "3D court"}</h2>
        <div className="row" style={{ gap: "0.5rem" }}>
          {splatUrl ? (
            <button
              type="button"
              className={`toggle-chip${showSplat ? " active" : ""}`}
              onClick={() => setShowSplat((v) => !v)}
              title="Toggle Gaussian splat environment (Nerfstudio)"
            >
              {showSplat ? "Splat on" : "Splat off"}
            </button>
          ) : null}
          {splatUrl ? (
            <button
              type="button"
              className={`toggle-chip${showCourtMesh ? " active" : ""}`}
              onClick={() => setShowCourtMesh((v) => !v)}
              title="Toggle procedural FIVB court mesh"
            >
              {showCourtMesh ? "Court on" : "Court off"}
            </button>
          ) : null}
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
          <button
            type="button"
            className={`toggle-chip${showPlayers ? " active" : ""}`}
            onClick={() => setShowPlayers((v) => !v)}
            title="Toggle 3D player capsules"
          >
            Players {showPlayers ? "on" : "off"}
          </button>
          <button
            type="button"
            className={`toggle-chip ball-vballnet${showBallVballnet ? " active" : ""}`}
            onClick={() => setShowBallVballnet((v) => !v)}
            title="Toggle VballNet 3D ball (yellow/black)"
          >
            VballNet {showBallVballnet ? "on" : "off"}
          </button>
          <button
            type="button"
            className={`toggle-chip ball-yolo${showBallYolo ? " active" : ""}`}
            onClick={() => setShowBallYolo((v) => !v)}
            title={
              hasYoloBall
                ? "Toggle SetOptics YOLO 3D ball (cyan)"
                : "Re-run analysis after deploying track_ball_yolo"
            }
            disabled={!hasYoloBall}
          >
            YOLO {showBallYolo && hasYoloBall ? "on" : "off"}
          </button>
          <button
            type="button"
            className={`toggle-chip ball-wasb${showBallWasb ? " active" : ""}`}
            onClick={() => setShowBallWasb((v) => !v)}
            title={
              hasWasbBall
                ? "Toggle WASB 3D ball (magenta)"
                : "Re-run analysis after deploying track_ball_wasb"
            }
            disabled={!hasWasbBall}
          >
            WASB {showBallWasb && hasWasbBall ? "on" : "off"}
          </button>
          <span className="meta-line">
            {showPlayers ? markers.length : 0} players
            {showBallVballnet && ballPos ? " + VballNet" : ""}
            {showBallYolo && ballYoloPos ? " + YOLO" : ""}
            {showBallWasb && ballWasbPos ? " + WASB" : ""} @{" "}
            {currentTime.toFixed(2)}s
            {!matchView ? " · drag to orbit" : ""}
          </span>
        </div>
      </div>
      <div
        className="video-shell court3d-canvas"
        style={{
          cursor: matchView ? "default" : "grab",
        }}
      >
        <Canvas
          shadows
          camera={{ position: defaultPos, fov: defaultFov }}
          style={{ width: "100%", height: "100%" }}
        >
          <ambientLight intensity={0.75} />
          <directionalLight position={[12, 20, 8]} intensity={1.1} castShadow />
          {showCourtMesh || !splatUrl ? (
            <CourtMesh length={length} width={width} />
          ) : null}
          {splatUrl && showSplat ? <GaussianSplat url={splatUrl} /> : null}
          {showPlayers ? (
            <Players markers={markers} roster={tracks?.players} />
          ) : null}
          {showBallVballnet && ballPos ? (
            <Ball ball={ballPos} color="#f5c518" />
          ) : null}
          {showBallYolo && ballYoloPos ? (
            <Ball ball={ballYoloPos} color="#00c8ff" />
          ) : null}
          {showBallWasb && ballWasbPos ? (
            <Ball ball={ballWasbPos} color="#ff5aa0" />
          ) : null}
          {cameraPose ? (
            <MatchedCamera pose={cameraPose} enabled={matchView} />
          ) : null}
          <OrbitControls
            makeDefault
            enabled={!matchView}
            enableRotate
            enablePan
            enableZoom
            target={orbitTarget}
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
