/** Shared domain types for Volleyball AI (Local v0). */

export type SourceType = "upload" | "youtube";

export type JobStatus =
  | "queued"
  | "running"
  | "needs_calibration"
  | "completed"
  | "failed";

export type PipelineStage =
  | "queued"
  | "fake"
  | "ingest"
  | "normalize"
  | "calibrate"
  | "track_players"
  | "track_ball"
  | "actions"
  | "score"
  | "project_3d"
  | "done";

export interface VideoMeta {
  duration_s?: number;
  fps?: number;
  width?: number;
  height?: number;
}

export interface Video {
  id: string;
  created_at: string;
  updated_at: string;
  name: string;
  source_type: SourceType;
  /** Original filename when uploaded */
  original_filename?: string;
  has_source: boolean;
  has_work: boolean;
  has_thumb: boolean;
  meta: VideoMeta;
}

export interface Job {
  id: string;
  video_id: string;
  status: JobStatus;
  stage: PipelineStage;
  /** 0–1 */
  progress: number;
  error?: string | null;
  retryable?: boolean;
  pipeline_version: string;
  cloud_run_execution_name?: string | null;
  created_at: string;
  updated_at: string;
}

export interface Point2 {
  x: number;
  y: number;
}

export interface CalibrationKeyframe {
  t: number;
  /** Image pixel corners (usually 4) */
  image_points: Point2[];
  /** Corresponding court points in meters */
  court_points_m: Point2[];
}

export interface CalibrationSegment {
  t0: number;
  t1: number;
  keyframe_index: number;
}

export interface CalibrationCamera {
  position: [number, number, number];
  R: number[];
  t: number[];
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  image_width: number;
  image_height: number;
  fov_y_deg: number;
}

export interface Calibration {
  video_id: string;
  pipeline_version: string;
  court: {
    length_m: number;
    width_m: number;
  };
  keyframes: CalibrationKeyframe[];
  segments?: CalibrationSegment[];
  /** Row-major 3x3 homography image→court when available */
  H?: number[] | null;
  /** Recovered camera pose for matched 3D view */
  camera?: CalibrationCamera | null;
}

export interface TrackFrame {
  t: number;
  bbox: [number, number, number, number];
  /** Body silhouette polygon in image pixels [[x,y], ...] from SAM mask */
  outline?: [number, number][];
  court_xy?: [number, number];
}

export interface PlayerTrack {
  track_id: number;
  frames: TrackFrame[];
}

export interface PlayersTracksFile {
  video_id: string;
  pipeline_version: string;
  players: PlayerTrack[];
  source?: "mock" | "sam3" | "sam3.1" | string;
}

export interface BallFrame {
  t: number;
  /** Image-space center + radius (pixels) when available */
  xy?: [number, number];
  r?: number;
  /** Court meters; z is height above court */
  court_xyz?: [number, number, number];
}

export interface BallTracksFile {
  video_id: string;
  pipeline_version: string;
  frames: BallFrame[];
  source?: "mock" | "modal" | string;
}

export const PIPELINE_VERSION = "0.1.0";

export const DEFAULT_COURT = {
  length_m: 18,
  width_m: 9,
} as const;
