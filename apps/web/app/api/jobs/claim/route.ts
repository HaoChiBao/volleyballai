import { NextResponse } from "next/server";
import { claimNextQueuedJob } from "@/lib/store";

export const runtime = "nodejs";

/** Local worker calls this to claim the next queued job. */
export async function POST() {
  const job = await claimNextQueuedJob();
  if (!job) {
    return NextResponse.json({ job: null });
  }
  return NextResponse.json({ job });
}
