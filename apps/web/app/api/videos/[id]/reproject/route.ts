import { NextResponse } from "next/server";
import { reprojectArtifacts } from "@/lib/reproject";
import { getCalibration, getVideo } from "@/lib/store";

export const runtime = "nodejs";

/** Recompute camera pose + court_xyz / court3d from existing calibration. */
export async function POST(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const video = await getVideo(id);
  if (!video) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  const cal = await getCalibration(id);
  const corners = cal?.keyframes?.[0]?.image_points?.length ?? 0;
  if (!cal || (!cal.H && corners < 4)) {
    return NextResponse.json(
      { error: "No calibration to reproject" },
      { status: 400 },
    );
  }

  await reprojectArtifacts(id, cal, {
    width: video.meta.width ?? cal.camera?.image_width ?? 1280,
    height: video.meta.height ?? cal.camera?.image_height ?? 720,
  });

  const next = await getCalibration(id);
  return NextResponse.json({ ok: true, calibration: next });
}
