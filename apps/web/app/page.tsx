import Link from "next/link";
import { getLatestRunPointer, listVideos, listJobsForVideo } from "@/lib/store";
import {
  formatRunDateTime,
  formatRunDuration,
  formatRunTiming,
} from "@/lib/formatRun";

export const dynamic = "force-dynamic";

export default async function LibraryPage() {
  const videos = await listVideos();

  return (
    <div className="stack">
      <div className="page-header">
        <h1>Library</h1>
        <p className="lede">
          Match clips and analysis runs. Upload a court video to start tracking.
        </p>
      </div>

      <div className="row">
        <Link className="button" href="/upload">
          Upload video
        </Link>
      </div>

      {videos.length === 0 ? (
        <div className="card empty-state">
          <strong>No videos yet</strong>
          Upload a short indoor clip to start your first analysis.
        </div>
      ) : (
        <div className="card-list">
          {await Promise.all(
            videos.map(async (video) => {
              const jobs = await listJobsForVideo(video.id);
              const latestJob = jobs[0];
              // CLI / direct pipeline runs write latest_run.json but may not
              // create a job — prefer the pointer so fresh analyzes show up.
              const pointer = await getLatestRunPointer(video.id);
              const run = pointer?.run_id
                ? {
                    run_id: pointer.run_id,
                    started_at: pointer.started_at,
                    finished_at: pointer.finished_at ?? null,
                    duration_s: pointer.duration_s ?? null,
                  }
                : (latestJob?.run ?? null);
              const status = pointer?.finished_at
                ? "completed"
                : latestJob?.status;
              const clipDur =
                video.meta.duration_s != null
                  ? `${video.meta.duration_s.toFixed(1)}s clip`
                  : null;
              return (
                <Link
                  key={video.id}
                  href={`/videos/${video.id}`}
                  className="card card-link"
                >
                  <div className="row between">
                    <strong>{video.name}</strong>
                    {status ? (
                      <span className={`badge ${status}`}>{status}</span>
                    ) : (
                      <span className="badge">no job</span>
                    )}
                  </div>
                  <div className="meta-line">
                    {video.original_filename ?? video.id}
                    {clipDur ? ` · ${clipDur}` : ""}
                  </div>
                  <div className="meta-line">
                    Uploaded {formatRunDateTime(video.created_at)}
                  </div>
                  {run?.started_at ? (
                    <div className="meta-line">
                      {run.finished_at ? (
                        <>
                          Analysis {formatRunDateTime(run.started_at)} →{" "}
                          {formatRunDateTime(run.finished_at)} ·{" "}
                          <strong>
                            {formatRunDuration(
                              run.started_at,
                              run.finished_at,
                              run.duration_s,
                            )}
                          </strong>
                        </>
                      ) : (
                        formatRunTiming(run)
                      )}
                      {run.run_id ? (
                        <span className="muted"> · {run.run_id}</span>
                      ) : null}
                    </div>
                  ) : latestJob ? (
                    <div className="meta-line">
                      Stage <code>{latestJob.stage}</code>{" "}
                      {(latestJob.progress * 100).toFixed(0)}%
                    </div>
                  ) : null}
                </Link>
              );
            }),
          )}
        </div>
      )}
    </div>
  );
}
