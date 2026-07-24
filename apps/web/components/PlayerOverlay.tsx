"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { PlayersTracksFile } from "@volleyballai/types";

const COLORS = ["#0b6e4f", "#1d4ed8", "#b45309", "#be123c", "#7c3aed", "#0f766e"];

export function PlayerOverlay({
  mediaUrl,
  tracks,
}: {
  mediaUrl: string;
  tracks: PlayersTracksFile | null;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [t, setT] = useState(0);
  const [size, setSize] = useState({ w: 1280, h: 720 });

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    const onTime = () => setT(el.currentTime);
    const onMeta = () =>
      setSize({ w: el.videoWidth || 1280, h: el.videoHeight || 720 });
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("seeked", onTime);
    el.addEventListener("loadedmetadata", onMeta);
    return () => {
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("seeked", onTime);
      el.removeEventListener("loadedmetadata", onMeta);
    };
  }, [mediaUrl]);

  const boxes = useMemo(() => {
    if (!tracks) return [];
    return tracks.players.map((p, idx) => {
      if (!p.frames.length) return null;
      const frame = p.frames.reduce((a, b) =>
        Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b,
      );
      if (Math.abs(frame.t - t) > 0.35) return null;
      return {
        track_id: p.track_id,
        bbox: frame.bbox,
        color: COLORS[idx % COLORS.length],
      };
    }).filter(Boolean) as {
      track_id: number;
      bbox: [number, number, number, number];
      color: string;
    }[];
  }, [tracks, t]);

  return (
    <div className="card stack">
      <div className="row between">
        <h2>Player overlay</h2>
        <span className="meta-line">
          {tracks
            ? `${tracks.players.length} tracks · t=${t.toFixed(2)}s`
            : "No tracks yet — run analysis"}
        </span>
      </div>
      <div className="video-shell" style={{ position: "relative" }}>
        <video ref={videoRef} src={mediaUrl} controls />
        <svg
          viewBox={`0 0 ${size.w} ${size.h}`}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
          }}
        >
          {boxes.map((b) => (
            <g key={b.track_id}>
              <rect
                x={b.bbox[0]}
                y={b.bbox[1]}
                width={b.bbox[2]}
                height={b.bbox[3]}
                fill="none"
                stroke={b.color}
                strokeWidth={Math.max(2, size.w / 400)}
              />
              <text
                x={b.bbox[0] + 4}
                y={b.bbox[1] - 6}
                fill={b.color}
                fontSize={Math.max(14, size.w / 60)}
                fontFamily="sans-serif"
                fontWeight="700"
              >
                #{b.track_id}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
