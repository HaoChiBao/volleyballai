import { NextResponse } from "next/server";
import type { CourtKeypointsFile } from "@volleyballai/types";
import { DEFAULT_COURT } from "@volleyballai/types";
import { calibrationFromCourtKeypoints } from "@/lib/courtFromKeypoints";
import { reprojectArtifacts } from "@/lib/reproject";
import {
  getCalibration,
  getCourtKeypoints,
  getLatestRunPointer,
  getVideo,
  saveCalibration,
} from "@/lib/store";

export const runtime = "nodejs";

/**
 * Build calibration + matched 3D camera from Modal court.keypoints.json.
 * Skips overwrite when an existing manual calibration is present, unless
 * `?force=1` is passed.
 */
export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const video = await getVideo(id);
  if (!video) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const url = new URL(request.url);
  const force = url.searchParams.get("force") === "1";

  const existing = await getCalibration(id);
  if (existing?.source === "manual" && !force) {
    return NextResponse.json(
      {
        error: "Manual calibration present; pass ?force=1 to overwrite",
        calibration: existing,
      },
      { status: 409 },
    );
  }

  const keypoints = (await getCourtKeypoints(id)) as CourtKeypointsFile | null;
  if (!keypoints?.frames?.length) {
    return NextResponse.json(
      { error: "No court.keypoints.json — run analysis first" },
      { status: 400 },
    );
  }

  let body: { length_m?: number; width_m?: number } = {};
  try {
    if (request.headers.get("content-type")?.includes("application/json")) {
      body = (await request.json()) as typeof body;
    }
  } catch {
    body = {};
  }

  const pointer = await getLatestRunPointer(id);
  const calibration = calibrationFromCourtKeypoints(keypoints, {
    videoId: id,
    length_m: body.length_m ?? existing?.court?.length_m ?? DEFAULT_COURT.length_m,
    width_m: body.width_m ?? existing?.court?.width_m ?? DEFAULT_COURT.width_m,
    imageWidth: keypoints.image_size?.width ?? video.meta.width,
    imageHeight: keypoints.image_size?.height ?? video.meta.height,
    fromRunId: pointer?.run_id ?? null,
  });

  if (!calibration?.H) {
    return NextResponse.json(
      {
        error:
          "Could not solve homography from keypoints (need ≥4 visible landmarks)",
      },
      { status: 422 },
    );
  }

  await saveCalibration(id, calibration);
  await reprojectArtifacts(id, calibration, {
    width: keypoints.image_size?.width ?? video.meta.width ?? 1280,
    height: keypoints.image_size?.height ?? video.meta.height ?? 720,
  });

  const saved = await getCalibration(id);
  return NextResponse.json({
    calibration: saved ?? calibration,
    source: "auto_keypoints",
  });
}
