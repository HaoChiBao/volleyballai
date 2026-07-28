import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";
import { videoDir } from "@/lib/paths";
import { getVideo, resolveArtifactDir } from "@/lib/store";

export const runtime = "nodejs";

async function findSpatialDir(videoId: string): Promise<string | null> {
  const candidates = [
    path.join(videoDir(videoId), "spatial"),
    path.join(await resolveArtifactDir(videoId), "spatial"),
  ];
  for (const dir of candidates) {
    try {
      await fs.access(path.join(dir, "scene.ply"));
      return dir;
    } catch {
      /* try next */
    }
  }
  return null;
}

/** GET ?kind=meta|ply — spatial Gaussian splat artifacts */
export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const video = await getVideo(id);
  if (!video) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const url = new URL(request.url);
  const kind = url.searchParams.get("kind") ?? "meta";
  const dir = await findSpatialDir(id);

  if (kind === "meta") {
    if (!dir) {
      return NextResponse.json({ spatial: null });
    }
    let meta: unknown = null;
    try {
      const raw = await fs.readFile(path.join(dir, "meta.json"), "utf8");
      meta = JSON.parse(raw) as unknown;
    } catch {
      meta = { ok: true, ply: "scene.ply" };
    }
    return NextResponse.json({
      spatial: {
        ...(typeof meta === "object" && meta ? meta : {}),
        ply_url: `/api/videos/${id}/spatial?kind=ply`,
        available: true,
      },
    });
  }

  if (kind === "ply") {
    if (!dir) {
      return NextResponse.json({ error: "Spatial scene missing" }, { status: 404 });
    }
    const plyPath = path.join(dir, "scene.ply");
    const data = await fs.readFile(plyPath);
    return new NextResponse(data, {
      headers: {
        "Content-Type": "application/octet-stream",
        "Cache-Control": "no-store",
        "Content-Disposition": `inline; filename="scene.ply"`,
      },
    });
  }

  return NextResponse.json({ error: "Invalid kind" }, { status: 400 });
}
