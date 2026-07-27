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
  | "detect_court"
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

/** Models + timing for one analysis pass (written on each pipeline run). */
export interface AnalysisRunModels {
  players: string;
  ball: string;
  /** Court keypoint detector (e.g. yolo_court_keypoints) */
  court?: string | null;
  /** e.g. sam sample rate */
  players_fps?: number | null;
  ball_infer_mode?: string | null;
  ball_model_key?: string | null;
  court_detections?: number | null;
}

export interface AnalysisRunInfo {
  /**
   * Filesystem-safe id for this pass, e.g. `2026-07-27_04-30-59Z`.
   * Artifacts live under `videos/{id}/runs/{run_id}/`.
   */
  run_id?: string | null;
  /** ISO timestamp when this analysis run started (pipeline begin). */
  started_at: string;
  /** ISO timestamp when artifacts were written. */
  finished_at?: string | null;
  /** Wall-clock seconds from started_at → finished_at (set when finished). */
  duration_s?: number | null;
  /** Relative dir from video root, e.g. `runs/2026-07-27_04-30-59Z`. */
  relative_dir?: string | null;
  pipeline_version: string;
  mock?: boolean;
  models: AnalysisRunModels;
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
  /** Set when the worker starts / finishes an analysis pass. */
  run?: AnalysisRunInfo | null;
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

export type CalibrationSource = "manual" | "auto_keypoints";

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
  /**
   * Where correspondences came from.
   * `auto_keypoints` = YOLOv11n-pose court model; `manual` = UI line drawing.
   */
  source?: CalibrationSource | null;
  /** Run folder that produced auto keypoints, when source=auto_keypoints */
  from_run_id?: string | null;
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
  /** When this track file was produced + models used. */
  run?: AnalysisRunInfo;
  sam_fps?: number;
  model?: string;
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
  source?: "mock" | "modal" | "modal-motion" | "vballnet" | string;
  /** When this track file was produced + models used. */
  run?: AnalysisRunInfo;
  model?: string;
  model_key?: string;
  infer_mode?: string;
}

/** One of 14 court keypoints from volley-ref-ai YOLOv11n-pose. */
export interface CourtKeypoint {
  name: string;
  /** Image pixels; null when not visible / below conf */
  xy: [number, number] | null;
  conf: number;
  visible: boolean;
  /** Corresponding FIVB court point in meters */
  court_m: Point2;
}

export interface CourtKeypointsFrame {
  t: number;
  frame_index: number;
  bbox?: number[];
  box_conf?: number;
  keypoints: CourtKeypoint[];
}

/** Auto court detection artifact (`court.keypoints.json`). */
export interface CourtKeypointsFile {
  video_id: string;
  pipeline_version: string;
  source?: "volley-ref-ai" | string;
  model?: string;
  model_repo?: string;
  keypoint_names: string[];
  skeleton: [number, number][];
  court_points_m: Point2[];
  image_size?: { width: number; height: number };
  sample_fps?: number | null;
  frames: CourtKeypointsFrame[];
  detections?: number;
  run?: AnalysisRunInfo;
}

export const PIPELINE_VERSION = "0.1.0";

export const DEFAULT_COURT = {
  length_m: 18,
  width_m: 9,
} as const;
