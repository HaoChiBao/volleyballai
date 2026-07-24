"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo } from "react";
import { BoxGeometry, EdgesGeometry } from "three";

export type Court3dSample = {
  t: number;
  players: { track_id: number; x: number; y: number; z: number }[];
};

export type Court3dFile = {
  court: { length_m: number; width_m: number };
  samples: Court3dSample[];
};

const COLORS = ["#0b6e4f", "#1d4ed8", "#b45309", "#be123c", "#7c3aed", "#0f766e"];

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
        <meshStandardMaterial color="#c4a574" />
      </mesh>
      <mesh position={[length / 2, 0.02, width / 2]}>
        <boxGeometry args={[0.06, 0.02, width]} />
        <meshStandardMaterial color="#f5f5f5" />
      </mesh>
      <mesh position={[length / 2, 1.2, width / 2]}>
        <boxGeometry args={[0.04, 2.4, width]} />
        <meshStandardMaterial color="#222222" transparent opacity={0.35} />
      </mesh>
      <lineSegments
        geometry={edges}
        position={[length / 2, 0, width / 2]}
      >
        <lineBasicMaterial color="#ffffff" />
      </lineSegments>
    </group>
  );
}

function Players({ markers }: { markers: Court3dSample["players"] }) {
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

export function Court3D({
  court3d,
  currentTime,
}: {
  court3d: Court3dFile | null;
  currentTime: number;
}) {
  const sample = useMemo(() => {
    if (!court3d?.samples?.length) return null;
    return court3d.samples.reduce((a, b) =>
      Math.abs(a.t - currentTime) < Math.abs(b.t - currentTime) ? a : b,
    );
  }, [court3d, currentTime]);

  const length = court3d?.court.length_m ?? 18;
  const width = court3d?.court.width_m ?? 9;

  return (
    <div className="card stack">
      <div className="row between">
        <h2>3D court</h2>
        <span className="meta-line">
          {sample
            ? `${sample.players.length} players @ ${sample.t.toFixed(2)}s`
            : "Calibrate to project players onto the court"}
        </span>
      </div>
      <div
        className="video-shell"
        style={{ height: 360, background: "#dfe6ee" }}
      >
        <Canvas shadows camera={{ position: [9, 14, 22], fov: 42 }}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[12, 20, 8]} intensity={1.1} castShadow />
          <CourtMesh length={length} width={width} />
          {sample ? <Players markers={sample.players} /> : null}
          <OrbitControls makeDefault target={[length / 2, 0, width / 2]} />
        </Canvas>
      </div>
      <p className="hint">
        Drag to orbit. Positions sync with the video timeline after calibration.
      </p>
    </div>
  );
}
