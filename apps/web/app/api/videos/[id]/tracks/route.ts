import { NextResponse } from "next/server";
import { getPlayersTracks, getVideo } from "@/lib/store";

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
  const tracks = await getPlayersTracks(id);
  return NextResponse.json({ tracks });
}
