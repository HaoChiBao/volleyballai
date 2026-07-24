"use client";

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import type {
  BallTracksFile,
  Calibration,
  Job,
  PlayersTracksFile,
  Video,
} from "@volleyballai/types";
import { AnalysisPlayer } from "./AnalysisPlayer";
import { CalibrationEditor } from "./CalibrationEditor";
import type { Court3dFile } from "./Court3D";
import { JobPanel } from "@/app/videos/[id]/JobPanel";

const Court3D = dynamic(
  () => import("./Court3D").then((m) => m.Court3D),
  {
    ssr: false,
    loading: () => (
      <div className="court3d-pane stack">
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
  initialBall,
  initialCourt3d,
}: {
  video: Video;
  initialJobs: Job[];
  initialCalibration: Calibration | null;
  initialTracks: PlayersTracksFile | null;
  initialBall: BallTracksFile | null;
  initialCourt3d: Court3dFile | null;
}) {
  const router = useRouter();
  const [calibration, setCalibration] = useState(initialCalibration);
  const [tracks, setTracks] = useState(initialTracks);
  const [ball, setBall] = useState(initialBall);
  const [court3d, setCourt3d] = useState(initialCourt3d);
  const [currentTime, setCurrentTime] = useState(0);

  const playKind = video.has_work ? "work" : video.has_source ? "source" : null;
  const mediaUrl = playKind
    ? `/api/videos/${video.id}/media?kind=${playKind}`
    : null;

  const reloadArtifacts = useCallback(async () => {
    const [tRes, bRes, cRes, calRes] = await Promise.all([
      fetch(`/api/videos/${video.id}/tracks`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/ball`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/court3d`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/calibration`, { cache: "no-store" }),
    ]);
    if (tRes.ok) {
      const data = (await tRes.json()) as { tracks: PlayersTracksFile | null };
      setTracks(data.tracks);
    }
    if (bRes.ok) {
      const data = (await bRes.json()) as { ball: BallTracksFile | null };
      setBall(data.ball);
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

  // Upgrade older calibrations (H set, camera missing) so 3D view matches video.
  useEffect(() => {
    if (!calibration?.H || calibration.camera) return;
    let cancelled = false;
    void (async () => {
      const res = await fetch(`/api/videos/${video.id}/reproject`, {
        method: "POST",
      });
      if (!cancelled && res.ok) await reloadArtifacts();
    })();
    return () => {
      cancelled = true;
    };
  }, [calibration?.H, calibration?.camera, video.id, reloadArtifacts]);

  return (
    <div className="stack workspace">
      <div className="row meta-line">
        <span>Source {video.has_source ? "ready" : "missing"}</span>
        <span>Work {video.has_work ? "ready" : "pending"}</span>
        <span>Thumb {video.has_thumb ? "ready" : "pending"}</span>
        <span>Players {tracks?.players.length ?? 0}</span>
        <span>Ball frames {ball?.frames.length ?? 0}</span>
        <span>Cal {calibration?.H ? "set" : "needed"}</span>
        <span>Cam {calibration?.camera ? "matched" : "—"}</span>
      </div>

      <JobPanel
        videoId={video.id}
        initialJobs={initialJobs}
        onJobSettled={() => void reloadArtifacts()}
      />

      {mediaUrl ? (
        <>
          <section className="match-stage card">
            <div className="match-stage-head row between">
              <div>
                <h2>Synced view</h2>
                <p className="hint" style={{ margin: 0 }}>
                  Video and 3D court share the same camera angle from your
                  corner calibration. Ball position updates with playback.
                </p>
              </div>
            </div>
            <div className="match-stage-grid">
              <AnalysisPlayer
                compact
                mediaUrl={mediaUrl}
                posterUrl={
                  video.has_thumb
                    ? `/api/videos/${video.id}/media?kind=thumb`
                    : undefined
                }
                calibration={calibration}
                tracks={tracks}
                ball={ball}
                court3d={court3d}
                onTime={setCurrentTime}
              />
              <Court3D
                compact
                court3d={court3d}
                calibration={calibration}
                tracks={tracks}
                ball={ball}
                currentTime={currentTime}
              />
            </div>
          </section>

          <CalibrationEditor
            videoId={video.id}
            mediaUrl={mediaUrl}
            initial={calibration}
            onSaved={() => void reloadArtifacts()}
          />
        </>
      ) : (
        <div className="card">
          <p className="muted">No playable media yet — queue analysis first.</p>
        </div>
      )}
    </div>
  );
}
