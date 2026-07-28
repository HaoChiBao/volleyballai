"use client";

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import type {
  BallTracksFile,
  Calibration,
  CameraMotionFile,
  Job,
  NetTracksFile,
  PlayersTracksFile,
  Video,
} from "@volleyballai/types";
import { AnalysisPlayer } from "./AnalysisPlayer";
import { CalibrationEditor } from "./CalibrationEditor";
import type { Court3dFile } from "./Court3D";
import { JobPanel } from "@/app/videos/[id]/JobPanel";
import {
  formatRunDateTime,
  formatRunDuration,
  formatRunModels,
} from "@/lib/formatRun";

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
  const [cameraMotion, setCameraMotion] = useState<CameraMotionFile | null>(
    null,
  );
  const [netTracks, setNetTracks] = useState<NetTracksFile | null>(null);
  const [splatUrl, setSplatUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const triedAutoCal = useRef(false);

  const playKind = video.has_work ? "work" : video.has_source ? "source" : null;
  const mediaUrl = playKind
    ? `/api/videos/${video.id}/media?kind=${playKind}`
    : null;
  const latestRun =
    tracks?.run ?? ball?.run ?? initialJobs[0]?.run ?? null;

  const reloadArtifacts = useCallback(async () => {
    const [tRes, bRes, cRes, calRes, sRes, mRes, nRes] = await Promise.all([
      fetch(`/api/videos/${video.id}/tracks`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/ball`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/court3d`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/calibration`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/spatial?kind=meta`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/camera-motion`, { cache: "no-store" }),
      fetch(`/api/videos/${video.id}/net`, { cache: "no-store" }),
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
    if (sRes.ok) {
      const data = (await sRes.json()) as {
        spatial: { available?: boolean; ply_url?: string } | null;
      };
      setSplatUrl(
        data.spatial?.available && data.spatial.ply_url
          ? data.spatial.ply_url
          : null,
      );
    }
    if (mRes.ok) {
      const data = (await mRes.json()) as {
        cameraMotion: CameraMotionFile | null;
      };
      setCameraMotion(data.cameraMotion);
    }
    if (nRes.ok) {
      const data = (await nRes.json()) as { netTracks: NetTracksFile | null };
      setNetTracks(data.netTracks);
    }
    router.refresh();
  }, [video.id, router]);

  useEffect(() => {
    void (async () => {
      const [mRes, nRes] = await Promise.all([
        fetch(`/api/videos/${video.id}/camera-motion`, { cache: "no-store" }),
        fetch(`/api/videos/${video.id}/net`, { cache: "no-store" }),
      ]);
      if (mRes.ok) {
        const data = (await mRes.json()) as {
          cameraMotion: CameraMotionFile | null;
        };
        setCameraMotion(data.cameraMotion);
      }
      if (nRes.ok) {
        const data = (await nRes.json()) as { netTracks: NetTracksFile | null };
        setNetTracks(data.netTracks);
      }
    })();
  }, [video.id]);

  useEffect(() => {
    void (async () => {
      const sRes = await fetch(`/api/videos/${video.id}/spatial?kind=meta`, {
        cache: "no-store",
      });
      if (!sRes.ok) return;
      const data = (await sRes.json()) as {
        spatial: { available?: boolean; ply_url?: string } | null;
      };
      setSplatUrl(
        data.spatial?.available && data.spatial.ply_url
          ? data.spatial.ply_url
          : null,
      );
    })();
  }, [video.id]);

  // Prefer YOLO court keypoints for 3D when no manual / net-settle calibration exists.
  useEffect(() => {
    if (triedAutoCal.current) return;
    if (calibration?.source === "manual") return;
    if (calibration?.source === "net_settle") return;
    if (
      calibration?.source === "auto_keypoints" &&
      calibration.H &&
      calibration.camera
    ) {
      return;
    }
    triedAutoCal.current = true;
    let cancelled = false;
    void (async () => {
      const res = await fetch(`/api/videos/${video.id}/calibrate-auto`, {
        method: "POST",
      });
      if (!cancelled && res.ok) {
        await reloadArtifacts();
        return;
      }
      if (!cancelled && calibration?.H && !calibration.camera) {
        const r2 = await fetch(`/api/videos/${video.id}/reproject`, {
          method: "POST",
        });
        if (r2.ok) await reloadArtifacts();
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    calibration?.source,
    calibration?.H,
    calibration?.camera,
    video.id,
    reloadArtifacts,
  ]);

  return (
    <div className="stack workspace">
      <div className="row meta-line">
        <span>Source {video.has_source ? "ready" : "missing"}</span>
        <span>Work {video.has_work ? "ready" : "pending"}</span>
        <span>Thumb {video.has_thumb ? "ready" : "pending"}</span>
        <span>Players {tracks?.players.length ?? 0}</span>
        <span>Ball frames {ball?.frames.length ?? 0}</span>
        <span>
          Cal{" "}
          {calibration?.H
            ? calibration.source === "auto_keypoints"
              ? "auto (YOLO)"
              : calibration.source === "net_settle"
                ? "net settle"
                : calibration.source === "manual"
                  ? "manual"
                  : "set"
            : "needed"}
        </span>
        <span>Cam {calibration?.camera ? "matched" : "—"}</span>
        <span>
          Motion{" "}
          {cameraMotion
            ? `${cameraMotion.settle_points?.length ?? cameraMotion.summary?.num_settle_points ?? "—"} settles / ${cameraMotion.summary?.num_segments ?? cameraMotion.segments.length} segs`
            : "—"}
        </span>
        <span>
          Net{" "}
          {netTracks?.frames?.length
            ? `${netTracks.frames.length} settles`
            : "—"}
        </span>
      </div>
      {latestRun?.started_at ? (
        <div className="stack" style={{ gap: "0.15rem" }}>
          <div className="row meta-line">
            <span>
              Started{" "}
              <strong>{formatRunDateTime(latestRun.started_at)}</strong>
            </span>
            {latestRun.finished_at ? (
              <span>
                Finished{" "}
                <strong>{formatRunDateTime(latestRun.finished_at)}</strong>
              </span>
            ) : (
              <span>Finished —</span>
            )}
            <span>
              Duration{" "}
              <strong>
                {formatRunDuration(
                  latestRun.started_at,
                  latestRun.finished_at,
                  latestRun.duration_s,
                )}
              </strong>
            </span>
            {latestRun.run_id ? (
              <span>
                <code>runs/{latestRun.run_id}</code>
              </span>
            ) : null}
          </div>
          <div className="row meta-line">
            <span>{formatRunModels(latestRun)}</span>
          </div>
        </div>
      ) : null}

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
                  Video and 3D court share the camera from court keypoints
                  (YOLO) or your manual lines. Ball updates with playback.
                  {splatUrl
                    ? " Gaussian splat environment loaded (orbit the gym; live players/ball from tracks)."
                    : ""}
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
                cameraMotion={cameraMotion}
                netTracks={netTracks}
                onTime={setCurrentTime}
              />
              <Court3D
                compact
                court3d={court3d}
                calibration={calibration}
                tracks={tracks}
                ball={ball}
                netTracks={netTracks}
                currentTime={currentTime}
                splatUrl={splatUrl}
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
