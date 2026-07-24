import { promises as fs } from "fs";
import { NextResponse } from "next/server";
import { filePathForVideoAsset, getVideo } from "@/lib/store";

export const runtime = "nodejs";

const KIND = new Set(["source", "work", "thumb"] as const);
type Kind = "source" | "work" | "thumb";

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
  const kind = (url.searchParams.get("kind") ?? "work") as Kind;
  if (!KIND.has(kind)) {
    return NextResponse.json({ error: "Invalid kind" }, { status: 400 });
  }

  const filePath = filePathForVideoAsset(id, kind);
  try {
    const data = await fs.readFile(filePath);
    const contentType =
      kind === "thumb" ? "image/jpeg" : "video/mp4";
    return new NextResponse(data, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json({ error: "File missing" }, { status: 404 });
  }
}
