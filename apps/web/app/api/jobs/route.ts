import { NextResponse } from "next/server";
import { createJob, listJobs, listJobsForVideo } from "@/lib/store";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const videoId = url.searchParams.get("video_id");
  const jobs = videoId
    ? await listJobsForVideo(videoId)
    : await listJobs();
  return NextResponse.json({ jobs });
}

export async function POST(request: Request) {
  let body: { video_id?: string };
  try {
    body = (await request.json()) as { video_id?: string };
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  if (!body.video_id) {
    return NextResponse.json({ error: "video_id required" }, { status: 400 });
  }

  try {
    const job = await createJob(body.video_id);
    return NextResponse.json(job, { status: 201 });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("not found") ? 404 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
