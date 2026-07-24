import { NextResponse } from "next/server";
import { createVideoFromUpload, listVideos } from "@/lib/store";

export const runtime = "nodejs";

export async function GET() {
  const videos = await listVideos();
  return NextResponse.json({ videos });
}

export async function POST(request: Request) {
  const form = await request.formData();
  const file = form.get("file");
  if (!(file instanceof File)) {
    return NextResponse.json({ error: "Missing file" }, { status: 400 });
  }

  const bytes = Buffer.from(await file.arrayBuffer());
  if (bytes.byteLength === 0) {
    return NextResponse.json({ error: "Empty file" }, { status: 400 });
  }

  const video = await createVideoFromUpload({
    filename: file.name || "upload.mp4",
    bytes,
  });

  return NextResponse.json(video, { status: 201 });
}
