import { NextResponse } from "next/server";
import type { PipelineStageTarget } from "@volleyballai/types";
import { PIPELINE_STAGE_TARGETS } from "@volleyballai/types";
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
  let body: { video_id?: string; stages?: string[] };
  try {
    body = (await request.json()) as { video_id?: string; stages?: string[] };
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  if (!body.video_id) {
    return NextResponse.json({ error: "video_id required" }, { status: 400 });
  }

  let stages: PipelineStageTarget[] | undefined;
  if (Array.isArray(body.stages) && body.stages.length > 0) {
    const allowed = new Set<string>(PIPELINE_STAGE_TARGETS);
    stages = body.stages.filter((s): s is PipelineStageTarget =>
      allowed.has(s),
    );
    if (stages.length === 0) {
      return NextResponse.json(
        {
          error: `Invalid stages (allowed: ${PIPELINE_STAGE_TARGETS.join(", ")})`,
        },
        { status: 400 },
      );
    }
  }

  try {
    const job = await createJob(body.video_id, { stages: stages ?? null });
    return NextResponse.json(job, { status: 201 });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("not found") ? 404 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
