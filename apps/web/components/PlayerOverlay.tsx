"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { PlayersTracksFile, TrackFrame } from "@volleyballai/types";
import { bracketFrames, lerpBbox } from "@/lib/trackInterp";
import { playerTrackScale, scaleBbox } from "@/lib/trackScale";

const COLORS = [
  "#ff5a36",
  "#2dd4bf",
  "#3b82f6",
  "#f59e0b",
  "#a855f7",
  "#22c55e",
  "#ec4899",
  "#06b6d4",
];
const PLAYER_MAX_GAP_S = 1.25;

function hexToRgba(hex: string, alpha: number) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

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
    const syncSize = () => {
      if (el.videoWidth > 0 && el.videoHeight > 0) {
        setSize({ w: el.videoWidth, h: el.videoHeight });
      }
    };
    syncSize();
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("seeked", onTime);
    el.addEventListener("loadedmetadata", syncSize);
    el.addEventListener("loadeddata", syncSize);
    return () => {
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("seeked", onTime);
      el.removeEventListener("loadedmetadata", syncSize);
      el.removeEventListener("loadeddata", syncSize);
    };
  }, [mediaUrl]);

  const boxes = useMemo(() => {
    if (!tracks) return [];
    const scale = playerTrackScale(tracks, size.w, size.h);
    return tracks.players.map((p, idx) => {
      if (!p.frames.length) return null;
      const br = bracketFrames(p.frames, t, PLAYER_MAX_GAP_S);
      if (!br) return null;
      const bbox =
        br.kind === "lerp"
          ? scaleBbox(
              lerpBbox(
                (br.a as TrackFrame).bbox,
                (br.b as TrackFrame).bbox,
                br.u,
              ),
              scale,
            )
          : scaleBbox((br.frame as TrackFrame).bbox, scale);
      return {
        track_id: p.track_id,
        label: idx + 1,
        bbox,
        color: COLORS[idx % COLORS.length],
      };
    }).filter(Boolean) as {
      track_id: number;
      label: number;
      bbox: [number, number, number, number];
      color: string;
    }[];
  }, [tracks, t, size.w, size.h]);

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
      <div className="video-shell analysis-shell">
        <div className="analysis-video-box">
          <video ref={videoRef} src={mediaUrl} controls />
          <svg
            className="analysis-overlay"
            width="100%"
            height="100%"
            viewBox={`0 0 ${size.w} ${size.h}`}
            preserveAspectRatio="none"
          >
            {boxes.map((b) => (
              <g key={b.track_id}>
                <rect
                  x={b.bbox[0]}
                  y={b.bbox[1]}
                  width={b.bbox[2]}
                  height={b.bbox[3]}
                  fill={hexToRgba(b.color, 0.35)}
                  stroke="none"
                />
                <text
                  x={b.bbox[0] + b.bbox[2] / 2}
                  y={b.bbox[1] - 6}
                  fill={b.color}
                  stroke="rgba(0,0,0,0.55)"
                  strokeWidth={Math.max(2, size.w / 500)}
                  paintOrder="stroke"
                  fontSize={Math.max(14, size.w / 60)}
                  fontFamily="sans-serif"
                  fontWeight="700"
                  textAnchor="middle"
                >
                  #{b.label}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
      {tracks?.players.length ? (
        <div className="player-playlist" aria-label="Player playlist">
          <div className="player-playlist-head row between">
            <span className="meta-line">Players</span>
            <span className="meta-line">{tracks.players.length} tracked</span>
          </div>
          <ul className="player-playlist-list">
            {tracks.players.map((p, idx) => {
              const color = COLORS[idx % COLORS.length];
              const visible = boxes.some((b) => b.track_id === p.track_id);
              return (
                <li
                  key={p.track_id}
                  className={`player-playlist-item${visible ? " visible" : ""}`}
                >
                  <span
                    className="player-playlist-swatch"
                    style={{ background: hexToRgba(color, 0.55) }}
                    aria-hidden
                  />
                  <span className="player-playlist-label" style={{ color }}>
                    #{idx + 1}
                  </span>
                  <span className="player-playlist-status meta-line">
                    {visible ? "in frame" : "off"}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
