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

  let body: Partial<Calibration>;
  try {
    body = (await request.json()) as Partial<Calibration>;
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const keyframes = body.keyframes ?? [];
  if (!keyframes[0] || keyframes[0].image_points.length < 4) {
    return NextResponse.json(
      { error: "Need 4 image corner points" },
      { status: 400 },
    );
  }

  let calibration: Calibration = {
    video_id: id,
    pipeline_version: PIPELINE_VERSION,
    court: body.court ?? { ...DEFAULT_COURT },
    keyframes,
    segments: body.segments,
    H: body.H ?? null,
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
  await reprojectArtifacts(id, calibration);

  return NextResponse.json({ calibration });
}
