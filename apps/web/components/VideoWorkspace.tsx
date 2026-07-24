"use client";

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import type { Calibration, Job, PlayersTracksFile, Video } from "@volleyballai/types";
import { CalibrationEditor } from "./CalibrationEditor";
import { PlayerOverlay } from "./PlayerOverlay";
import type { Court3dFile } from "./Court3D";
import { JobPanel } from "@/app/videos/[id]/JobPanel";

const Court3D = dynamic(
  () => import("./Court3D").then((m) => m.Court3D),
  {
    ssr: false,
    loading: () => (
      <div className="card">
        <p className="muted">Loading 3D court…</p>
      </div>
    ),
  },
);

export function VideoWorkspace({
  video,
  initialJobs,
  initialCalibration,
  initialTracks,
  initialCourt3d,
}: {
  video: Video;
  initialJobs: Job[];
  initialCalibration: Calibration | null;
  initialTracks: PlayersTracksFile | null;
  initialCourt3d: Court3dFile | null;
}) {
  const router = useRouter();
  const [calibration, setCalibration] = useState(initialCalibration);
  const [tracks, setTracks] = useState(initialTracks);
  const [court3d, setCourt3d] = useState(initialCourt3d);
  const [currentTime, setCurrentTime] = useState(0);

  const playKind = video.has_work ? "work" : video.has_source ? "source" : null;
  const mediaUrl = playKind
    ? `/api/videos/${video.id}/media?kind=${playKind}`
    : null;

  const reloadArtifacts = useCallback(async () => {
    const [tRes, cRes, calRes] = await Promise.all([
      fetch(`/api/videos/${video.id}/tracks`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/court3d`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/calibration`, { cache: "no-store" }),
    ]);
    if (tRes.ok) {
      const data = (await tRes.json()) as { tracks: PlayersTracksFile | null };
      setTracks(data.tracks);
    }
    if (cRes.ok) {
      const data = (await cRes.json()) as { court3d: Court3dFile | null };
      setCourt3d(data.court3d);
    }
    if (calRes.ok) {
      const data = (await calRes.json()) as {
        calibration: Calibration | null;
      };
      setCalibration(data.calibration);
    }
    router.refresh();
  }, [video.id, router]);

  return (
    <div className="stack workspace">
      <div className="card stack">
        <h2>Preview</h2>
        {mediaUrl ? (
          <div className="video-shell">
            <video
              id="main-preview"
              controls
              src={mediaUrl}
              poster={
                video.has_thumb
                  ? `/api/videos/${video.id}/media?kind=thumb`
                  : undefined
              }
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onSeeked={(e) => setCurrentTime(e.currentTarget.currentTime)}
            />
          </div>
        ) : (
          <p className="muted">No playable media yet.</p>
        )}
        <div className="row meta-line">
          <span>Source {video.has_source ? "ready" : "missing"}</span>
          <span>Work {video.has_work ? "ready" : "pending"}</span>
          <span>Thumb {video.has_thumb ? "ready" : "pending"}</span>
          <span>Tracks {tracks?.players.length ?? 0}</span>
          <span>Cal {calibration?.H ? "set" : "needed"}</span>
        </div>
      </div>

      <JobPanel
        videoId={video.id}
        initialJobs={initialJobs}
        onJobSettled={() => void reloadArtifacts()}
      />

      {mediaUrl ? (
        <>
          <CalibrationEditor
            videoId={video.id}
            mediaUrl={mediaUrl}
            initial={calibration}
            onSaved={() => void reloadArtifacts()}
          />
          <PlayerOverlay mediaUrl={mediaUrl} tracks={tracks} />
          <Court3D court3d={court3d} currentTime={currentTime} />
        </>
      ) : null}
    </div>
  );
}
