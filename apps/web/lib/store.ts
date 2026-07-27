import { promises as fs } from "fs";
import path from "path";
import { randomUUID } from "crypto";
import type {
  BallTracksFile,
  Calibration,
  Job,
  PipelineStage,
  PlayersTracksFile,
  Video,
  VideoMeta,
} from "@volleyballai/types";
import { DEFAULT_COURT, PIPELINE_VERSION } from "@volleyballai/types";
import {
  dataRoot,
  jobsPath,
  latestRunPointerPath,
  videoDir,
  videoMetaPath,
  videoRunDir,
  videosRoot,
} from "./paths";

export type LatestRunPointer = {
  run_id: string;
  started_at?: string;
  finished_at?: string | null;
  duration_s?: number | null;
  relative_dir?: string;
  pipeline_version?: string;
};

async function ensureDataDirs(): Promise<void> {
  await fs.mkdir(videosRoot(), { recursive: true });
  await fs.mkdir(dataRoot(), { recursive: true });
  try {
    await fs.access(jobsPath());
  } catch {
    await fs.writeFile(jobsPath(), "[]\n", "utf8");
  }
}

async function readJsonFile<T>(filePath: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

async function writeJsonFile(filePath: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tmp = `${filePath}.${process.pid}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(value, null, 2) + "\n", "utf8");
  await fs.rename(tmp, filePath);
}

function nowIso(): string {
  return new Date().toISOString();
}

export async function listVideos(): Promise<Video[]> {
  await ensureDataDirs();
  const entries = await fs.readdir(videosRoot(), { withFileTypes: true });
  const videos: Video[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const video = await getVideo(entry.name);
    if (video) videos.push(video);
  }
  videos.sort((a, b) => b.created_at.localeCompare(a.created_at));
  return videos;
}

export async function getVideo(id: string): Promise<Video | null> {
  await ensureDataDirs();
  const metaPath = videoMetaPath(id);
  const stored = await readJsonFile<Partial<Video> | null>(metaPath, null);
  if (!stored || !stored.id) return null;

  const dir = videoDir(id);
  const has = async (name: string) => {
    try {
      await fs.access(path.join(dir, name));
      return true;
    } catch {
      return false;
    }
  };

  return {
    id: stored.id,
    created_at: stored.created_at ?? nowIso(),
    updated_at: stored.updated_at ?? stored.created_at ?? nowIso(),
    name: stored.name ?? id,
    source_type: stored.source_type ?? "upload",
    original_filename: stored.original_filename,
    has_source: await has("source.mp4"),
    has_work: await has("work.mp4"),
    has_thumb: await has("thumb.jpg"),
    meta: stored.meta ?? {},
  };
}

export async function createVideoFromUpload(opts: {
  filename: string;
  bytes: Buffer;
}): Promise<Video> {
  await ensureDataDirs();
  const id = randomUUID();
  const dir = videoDir(id);
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(path.join(dir, "source.mp4"), opts.bytes);

  const created_at = nowIso();
  const video: Video = {
    id,
    created_at,
    updated_at: created_at,
    name: opts.filename.replace(/\.[^.]+$/, "") || id,
    source_type: "upload",
    original_filename: opts.filename,
    has_source: true,
    has_work: false,
    has_thumb: false,
    meta: {},
  };
  await writeJsonFile(videoMetaPath(id), video);
  return video;
}

export async function updateVideoMeta(
  id: string,
  patch: Partial<Pick<Video, "name" | "meta" | "updated_at">>,
): Promise<Video | null> {
  const existing = await getVideo(id);
  if (!existing) return null;
  const next: Video = {
    ...existing,
    ...patch,
    meta: { ...existing.meta, ...(patch.meta ?? {}) },
    updated_at: nowIso(),
  };
  await writeJsonFile(videoMetaPath(id), next);
  return getVideo(id);
}

export async function listJobs(): Promise<Job[]> {
  await ensureDataDirs();
  const jobs = await readJsonFile<Job[]>(jobsPath(), []);
  return jobs.sort((a, b) => b.created_at.localeCompare(a.created_at));
}

export async function getJob(id: string): Promise<Job | null> {
  const jobs = await listJobs();
  return jobs.find((j) => j.id === id) ?? null;
}

export async function listJobsForVideo(videoId: string): Promise<Job[]> {
  const jobs = await listJobs();
  return jobs.filter((j) => j.video_id === videoId);
}

export async function createJob(videoId: string): Promise<Job> {
  await ensureDataDirs();
  const video = await getVideo(videoId);
  if (!video) {
    throw new Error(`Video not found: ${videoId}`);
  }
  const jobs = await readJsonFile<Job[]>(jobsPath(), []);
  const created_at = nowIso();
  const job: Job = {
    id: randomUUID(),
    video_id: videoId,
    status: "queued",
    stage: "queued",
    progress: 0,
    error: null,
    retryable: true,
    pipeline_version: PIPELINE_VERSION,
    cloud_run_execution_name: null,
    created_at,
    updated_at: created_at,
  };
  jobs.push(job);
  await writeJsonFile(jobsPath(), jobs);
  return job;
}

export async function updateJob(
  id: string,
  patch: Partial<
    Pick<
      Job,
      | "status"
      | "stage"
      | "progress"
      | "error"
      | "retryable"
      | "cloud_run_execution_name"
      | "run"
    >
  >,
): Promise<Job | null> {
  await ensureDataDirs();
  const jobs = await readJsonFile<Job[]>(jobsPath(), []);
  const idx = jobs.findIndex((j) => j.id === id);
  if (idx < 0) return null;
  jobs[idx] = {
    ...jobs[idx],
    ...patch,
    updated_at: nowIso(),
  };
  await writeJsonFile(jobsPath(), jobs);
  return jobs[idx];
}

const STALE_RUNNING_MS = Number(process.env.JOB_STALE_MS ?? 30_000);

function isStaleRunning(job: Job): boolean {
  if (job.status !== "running") return false;
  const updated = Date.parse(job.updated_at);
  if (Number.isNaN(updated)) return true;
  return Date.now() - updated > STALE_RUNNING_MS;
}

export async function claimNextQueuedJob(): Promise<Job | null> {
  await ensureDataDirs();
  const jobs = await readJsonFile<Job[]>(jobsPath(), []);
  const idx = jobs.findIndex(
    (j) => j.status === "queued" || isStaleRunning(j),
  );
  if (idx < 0) return null;
  const claimedAt = nowIso();
  jobs[idx] = {
    ...jobs[idx],
    status: "running",
    stage: "ingest" satisfies PipelineStage,
    progress: 0.01,
    error: null,
    updated_at: claimedAt,
    run: {
      run_id: null,
      started_at: claimedAt,
      finished_at: null,
      duration_s: null,
      relative_dir: null,
      pipeline_version: jobs[idx].pipeline_version || PIPELINE_VERSION,
      models: {
        players: "pending",
        ball: "pending",
        court: "pending",
      },
    },
  };
  await writeJsonFile(jobsPath(), jobs);
  return jobs[idx];
}

export function filePathForVideoAsset(
  videoId: string,
  kind: "source" | "work" | "thumb",
): string {
  const names = {
    source: "source.mp4",
    work: "work.mp4",
    thumb: "thumb.jpg",
  } as const;
  return path.join(videoDir(videoId), names[kind]);
}

export async function getLatestRunPointer(
  videoId: string,
): Promise<LatestRunPointer | null> {
  return readJsonFile<LatestRunPointer | null>(
    latestRunPointerPath(videoId),
    null,
  );
}

/**
 * Resolve the directory that holds the current analysis artifacts.
 * Prefers `runs/{run_id}/` via latest_run.json; falls back to video root
 * for older layouts that wrote files flat.
 */
export async function resolveArtifactDir(videoId: string): Promise<string> {
  const pointer = await getLatestRunPointer(videoId);
  if (pointer?.run_id) {
    const dir = videoRunDir(videoId, pointer.run_id);
    try {
      await fs.access(dir);
      return dir;
    } catch {
      /* fall through */
    }
  }
  return videoDir(videoId);
}

async function readArtifactJson<T>(
  videoId: string,
  filename: string,
): Promise<T | null> {
  const artifactDir = await resolveArtifactDir(videoId);
  const primary = path.join(artifactDir, filename);
  const fromPrimary = await readJsonFile<T | null>(primary, null);
  if (fromPrimary) return fromPrimary;
  // Legacy flat layout / mid-migration
  if (artifactDir !== videoDir(videoId)) {
    return readJsonFile<T | null>(path.join(videoDir(videoId), filename), null);
  }
  return null;
}

export async function getCalibration(
  videoId: string,
): Promise<Calibration | null> {
  // Calibration stays video-scoped (shared across runs).
  return readJsonFile<Calibration | null>(
    path.join(videoDir(videoId), "calibration.json"),
    null,
  );
}

export async function saveCalibration(
  videoId: string,
  calibration: Calibration,
): Promise<Calibration> {
  const video = await getVideo(videoId);
  if (!video) throw new Error(`Video not found: ${videoId}`);
  const payload: Calibration = {
    ...calibration,
    video_id: videoId,
    pipeline_version: calibration.pipeline_version || PIPELINE_VERSION,
    court: calibration.court ?? { ...DEFAULT_COURT },
    source: calibration.source ?? null,
    from_run_id: calibration.from_run_id ?? null,
  };
  await writeJsonFile(path.join(videoDir(videoId), "calibration.json"), payload);
  return payload;
}

export async function getPlayersTracks(
  videoId: string,
): Promise<PlayersTracksFile | null> {
  return readArtifactJson<PlayersTracksFile>(videoId, "players.tracks.json");
}

export async function getCourt3d(videoId: string): Promise<unknown | null> {
  return readArtifactJson<unknown>(videoId, "court3d.json");
}

export async function getBallTracks(
  videoId: string,
): Promise<BallTracksFile | null> {
  return readArtifactJson<BallTracksFile>(videoId, "ball.tracks.json");
}

export async function getCourtKeypoints(
  videoId: string,
): Promise<unknown | null> {
  return readArtifactJson<unknown>(videoId, "court.keypoints.json");
}

export async function getAnalysisRun(
  videoId: string,
): Promise<import("@volleyballai/types").AnalysisRunInfo | null> {
  return readArtifactJson(videoId, "analysis.run.json");
}

export type { VideoMeta };
