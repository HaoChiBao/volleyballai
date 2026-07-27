import { NextResponse } from "next/server";
import type { Calibration } from "@volleyballai/types";
import { PIPELINE_VERSION, DEFAULT_COURT } from "@volleyballai/types";
import { ensureHomography, reprojectArtifacts } from "@/lib/reproject";
import { getCalibration, getVideo, saveCalibration } from "@/lib/store";

export const runtime = "nodejs";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const video = await getVideo(id);
  if (!video) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  const cal = await getCalibration(id);
  return NextResponse.json({ calibration: cal });
}

export async function PUT(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const video = await getVideo(id);
  if (!video) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  let body: Partial<Calibration> & {
    image_width?: number;
    image_height?: number;
  };
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const keyframes = body.keyframes ?? [];
  if (!keyframes[0] || keyframes[0].image_points.length < 4) {
    return NextResponse.json(
      { error: "Need at least 4 points from drawn court lines" },
      { status: 400 },
    );
  }

  const imageWidth =
    body.image_width ??
    video.meta.width ??
    body.camera?.image_width ??
    1280;
  const imageHeight =
    body.image_height ??
    video.meta.height ??
    body.camera?.image_height ??
    720;

  let calibration: Calibration = {
    video_id: id,
    pipeline_version: PIPELINE_VERSION,
    court: body.court ?? { ...DEFAULT_COURT },
    keyframes,
    segments: body.segments,
    H: body.H ?? null,
    // UI line drawing always wins over auto keypoints.
    source: "manual",
    from_run_id: null,
  };

  try {
    calibration = ensureHomography(calibration);
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Homography failed" },
      { status: 400 },
    );
  }

  await saveCalibration(id, calibration);
  await reprojectArtifacts(id, calibration, {
    width: imageWidth,
    height: imageHeight,
  });

  const saved = await getCalibration(id);
  return NextResponse.json({ calibration: saved ?? calibration });
}
