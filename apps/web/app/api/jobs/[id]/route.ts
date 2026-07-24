import { NextResponse } from "next/server";
import { getJob, updateJob } from "@/lib/store";

export const runtime = "nodejs";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const job = await getJob(id);
  if (!job) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  return NextResponse.json(job);
}

/** Used by the local worker to update progress. */
export async function PATCH(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  let body: Record<string, unknown>;
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const patch: Parameters<typeof updateJob>[1] = {};
  if (typeof body.status === "string") patch.status = body.status as never;
  if (typeof body.stage === "string") patch.stage = body.stage as never;
  if (typeof body.progress === "number") patch.progress = body.progress;
  if ("error" in body) patch.error = (body.error as string | null) ?? null;
  if (typeof body.retryable === "boolean") patch.retryable = body.retryable;
  if ("cloud_run_execution_name" in body) {
    patch.cloud_run_execution_name =
      (body.cloud_run_execution_name as string | null) ?? null;
  }

  const job = await updateJob(id, patch);

  if (!job) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  return NextResponse.json(job);
}
