import path from "path";

/** Repo root (volleyballai/) relative to apps/web */
export function repoRoot(): string {
  return path.resolve(process.cwd(), "../..");
}

export function dataRoot(): string {
  const fromEnv = process.env.DATA_DIR;
  if (fromEnv) {
    return path.isAbsolute(fromEnv)
      ? fromEnv
      : path.resolve(repoRoot(), fromEnv);
  }
  return path.resolve(repoRoot(), ".data");
}

export function jobsPath(): string {
  return path.join(dataRoot(), "jobs.json");
}

export function videosRoot(): string {
  return path.join(dataRoot(), "videos");
}

export function videoDir(videoId: string): string {
  return path.join(videosRoot(), videoId);
}

export function videoMetaPath(videoId: string): string {
  return path.join(videoDir(videoId), "meta.json");
}

export function latestRunPointerPath(videoId: string): string {
  return path.join(videoDir(videoId), "latest_run.json");
}

export function videoRunsRoot(videoId: string): string {
  return path.join(videoDir(videoId), "runs");
}

export function videoRunDir(videoId: string, runId: string): string {
  return path.join(videoRunsRoot(videoId), runId);
}
