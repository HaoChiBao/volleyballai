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

export type CalibrationSource = "manual" | "auto_keypoints" | "net_settle";

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

/** Gaussian splat environment from Modal Nerfstudio (`spatial/meta.json`). */
export interface SpatialSceneMeta {
  ok?: boolean;
  method?: string;
  max_iters?: number;
  num_frames_target?: number;
  appearance_embedding?: boolean;
  transient_burn_touches?: number;
  ply_bytes?: number;
  elapsed_s?: number;
  video_id?: string;
  volume?: string;
  local_ply?: string;
  downloaded_bytes?: number;
  note?: string;
  available?: boolean;
  ply_url?: string;
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

/** Camera-motion event for sparse net re-detect / timeline ticks. */
export type CameraMotionEventType =
  | "motion_start"
  | "motion_peak"
  | "motion_end";

export interface CameraMotionEvent {
  type: CameraMotionEventType;
  t: number;
  frame_index?: number;
  score?: number;
  /** Present on motion_end — pipeline should refresh net here. */
  suggest_net_redetect?: boolean;
}

export interface CameraMotionSegment {
  start_t: number;
  end_t: number;
  peak_t: number;
  peak_score: number;
  start_frame: number;
  end_frame: number;
}

export interface CameraMotionSettlePoint {
  t: number;
  frame_index?: number;
  /** motion_settled = after a pan; static_open = no motion in whole clip */
  kind: "motion_settled" | "static_open" | string;
  use_for_net_detect?: boolean;
}

export interface CameraMotionSettlePolicy {
  /** True when the first motion starts near t=0 — no reliable pose yet. */
  starts_unsettled: boolean;
  start_unsettled_s?: number;
  first_settle_t?: number | null;
  /** If starts_unsettled, apply this settle's pose for [0, first_settle_t]. */
  prefix_use_settle_t?: number | null;
  note?: string;
}

/**
 * Per-video camera motion artifact (`camera_motion.json`).
 * Produced by `worker.test_camera_motion` / future pipeline stage.
 * Samples omitted from the web payload to keep the file small.
 */
export interface CameraMotionFile {
  video_id: string;
  pipeline_version: string;
  source?: "camera_motion_test" | "pipeline" | string;
  method: "global_affine" | "phase_correlate" | "flow_median" | string;
  duration_s?: number;
  fps?: number;
  sample_fps?: number;
  analyze_max_side?: number;
  thresholds?: Record<string, number>;
  /** Merged motion segments (gaps ≤ merge_gap_s collapsed). */
  segments: CameraMotionSegment[];
  events: CameraMotionEvent[];
  /** Times the camera has settled — primary ticks for net / pose refresh. */
  settle_points?: CameraMotionSettlePoint[];
  /** Settles + static_refresh samples (~2× density) for net redetect. */
  net_sample_points?: CameraMotionSettlePoint[];
  settle_policy?: CameraMotionSettlePolicy;
  summary?: {
    num_segments?: number;
    num_segments_raw?: number;
    num_settle_points?: number;
    num_net_samples?: number;
    time_moving_s?: number;
    starts_unsettled?: boolean;
    recommend?: string;
  };
}

/** One settle-time net detection + FIVB PnP camera. */
export interface NetTrackFrame {
  t: number;
  frame_index?: number;
  trigger?: "settle" | string;
  kind?: string;
  net: {
    top_left: Point2;
    top_right: Point2;
    bottom_right: Point2;
    bottom_left: Point2;
  };
  camera?: CalibrationCamera | null;
  H?: number[] | null;
  reproj_err_px?: number;
  score?: number;
  mapping?: string;
  ground_lines?: {
    boundary?: [number, number][];
    center?: [number, number][];
    attack_a?: [number, number][];
    attack_b?: [number, number][];
  };
  model?: string;
  max_side?: number | null;
}

/** Per-video net tracks from settle → net detect (`net.tracks.json`). */
export interface NetTracksFile {
  video_id: string;
  pipeline_version: string;
  source?: "openai_net_settle" | string;
  model?: string;
  max_side?: number | null;
  fivb?: Record<string, number>;
  settle_policy?: CameraMotionSettlePolicy;
  primary_t?: number;
  frames: NetTrackFrame[];
  summary?: {
    num_settles?: number;
    primary_t?: number;
    primary_score?: number;
    primary_reproj_err_px?: number;
    starts_unsettled?: boolean;
  };
}

export const PIPELINE_VERSION = "0.1.0";

export const DEFAULT_COURT = {
  length_m: 18,
  width_m: 9,
} as const;
