"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { PlayersTracksFile, TrackFrame } from "@volleyballai/types";
import { bracketFrames, lerpBbox } from "@/lib/trackInterp";

const COLORS = ["#ffffff", "#d4d4d4", "#a3a3a3", "#737373", "#f5f5f5", "#e5e5e5"];
const PLAYER_MAX_GAP_S = 0.45;

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
      const br = bracketFrames(p.frames, t, PLAYER_MAX_GAP_S);
      if (!br) return null;
      const bbox =
        br.kind === "lerp"
          ? lerpBbox(
              (br.a as TrackFrame).bbox,
              (br.b as TrackFrame).bbox,
              br.u,
            )
          : (br.frame as TrackFrame).bbox;
      return {
        track_id: p.track_id,
        bbox,
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
