import { NextResponse } from "next/server";
import { getCameraMotion, getVideo } from "@/lib/store";

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
  const cameraMotion = await getCameraMotion(id);
  return NextResponse.json({ cameraMotion });
}
