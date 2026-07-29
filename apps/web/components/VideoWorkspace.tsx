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

/** Fetch that returns null on HTTP / network failure (never throws). */
async function fetchOk(
  url: string,
  init?: RequestInit,
): Promise<Response | null> {
  try {
    const res = await fetch(url, { cache: "no-store", ...init });
    return res.ok ? res : null;
  } catch {
    // Aborts, offline, HMR, extension-intercepted fetch, etc.
    return null;
  }
}

async function readJson<T>(res: Response | null): Promise<T | null> {
  if (!res) return null;
  try {
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export function VideoWorkspace({
  video,
  initialJobs,
  initialCalibration,
  initialTracks,
  initialBall,
  initialBallYolo,
  initialBallWasb,
  initialCourt3d,
}: {
  video: Video;
  initialJobs: Job[];
  initialCalibration: Calibration | null;
  initialTracks: PlayersTracksFile | null;
  initialBall: BallTracksFile | null;
  initialBallYolo: BallTracksFile | null;
  initialBallWasb: BallTracksFile | null;
  initialCourt3d: Court3dFile | null;
}) {
  const router = useRouter();
  const [calibration, setCalibration] = useState(initialCalibration);
  const [tracks, setTracks] = useState(initialTracks);
  const [ball, setBall] = useState(initialBall);
  const [ballYolo, setBallYolo] = useState(initialBallYolo);
  const [ballWasb, setBallWasb] = useState(initialBallWasb);
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
      fetchOk(`/api/videos/${video.id}/tracks`),
      fetchOk(`/api/videos/${video.id}/ball`),
      fetchOk(`/api/videos/${video.id}/court3d`),
      fetchOk(`/api/videos/${video.id}/calibration`),
      fetchOk(`/api/videos/${video.id}/spatial?kind=meta`),
      fetchOk(`/api/videos/${video.id}/camera-motion`),
      fetchOk(`/api/videos/${video.id}/net`),
    ]);
    const tracksData = await readJson<{ tracks: PlayersTracksFile | null }>(tRes);
    if (tracksData) setTracks(tracksData.tracks);
    const ballData = await readJson<{
      ball: BallTracksFile | null;
      ballYolo?: BallTracksFile | null;
      ballWasb?: BallTracksFile | null;
    }>(bRes);
    if (ballData) {
      setBall(ballData.ball);
      if ("ballYolo" in ballData) setBallYolo(ballData.ballYolo ?? null);
      if ("ballWasb" in ballData) setBallWasb(ballData.ballWasb ?? null);
    }
    const courtData = await readJson<{ court3d: Court3dFile | null }>(cRes);
    if (courtData) setCourt3d(courtData.court3d);
    const calData = await readJson<{ calibration: Calibration | null }>(calRes);
    if (calData) setCalibration(calData.calibration);
    const spatialData = await readJson<{
      spatial: { available?: boolean; ply_url?: string } | null;
    }>(sRes);
    if (spatialData) {
      setSplatUrl(
        spatialData.spatial?.available && spatialData.spatial.ply_url
          ? spatialData.spatial.ply_url
          : null,
      );
    }
    const motionData = await readJson<{
      cameraMotion: CameraMotionFile | null;
    }>(mRes);
    if (motionData) setCameraMotion(motionData.cameraMotion);
    const netData = await readJson<{ netTracks: NetTracksFile | null }>(nRes);
    if (netData) setNetTracks(netData.netTracks);
    router.refresh();
  }, [video.id, router]);

  // Optional overlays: camera motion, net, spatial meta (fail soft).
  useEffect(() => {
    const ac = new AbortController();
    void (async () => {
      const [mRes, nRes, sRes] = await Promise.all([
        fetchOk(`/api/videos/${video.id}/camera-motion`, {
          signal: ac.signal,
        }),
        fetchOk(`/api/videos/${video.id}/net`, { signal: ac.signal }),
        fetchOk(`/api/videos/${video.id}/spatial?kind=meta`, {
          signal: ac.signal,
        }),
      ]);
      if (ac.signal.aborted) return;

      const motionData = await readJson<{
        cameraMotion: CameraMotionFile | null;
      }>(mRes);
      if (motionData) setCameraMotion(motionData.cameraMotion);

      const netData = await readJson<{ netTracks: NetTracksFile | null }>(nRes);
      if (netData) setNetTracks(netData.netTracks);

      const spatialData = await readJson<{
        spatial: { available?: boolean; ply_url?: string } | null;
      }>(sRes);
      if (spatialData) {
        setSplatUrl(
          spatialData.spatial?.available && spatialData.spatial.ply_url
            ? spatialData.spatial.ply_url
            : null,
        );
      }
    })();
    return () => ac.abort();
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
    const ac = new AbortController();
    void (async () => {
      const res = await fetchOk(`/api/videos/${video.id}/calibrate-auto`, {
        method: "POST",
        signal: ac.signal,
      });
      if (ac.signal.aborted) return;
      if (res) {
        await reloadArtifacts();
        return;
      }
      if (calibration?.H && !calibration.camera) {
        const r2 = await fetchOk(`/api/videos/${video.id}/reproject`, {
          method: "POST",
          signal: ac.signal,
        });
        if (!ac.signal.aborted && r2) await reloadArtifacts();
      }
    })();
    return () => ac.abort();
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
        <span>YOLO ball {ballYolo?.frames.length ?? 0}</span>
        <span>WASB ball {ballWasb?.frames.length ?? 0}</span>
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
                ballYolo={ballYolo}
                ballWasb={ballWasb}
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
                ballYolo={ballYolo}
                ballWasb={ballWasb}
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
