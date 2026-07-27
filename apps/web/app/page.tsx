import Link from "next/link";
import { listVideos, listJobsForVideo } from "@/lib/store";
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
              const latest = jobs[0];
              const run = latest?.run ?? null;
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
                    {latest ? (
                      <span className={`badge ${latest.status}`}>
                        {latest.status}
                      </span>
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
                  ) : latest ? (
                    <div className="meta-line">
                      Stage <code>{latest.stage}</code>{" "}
                      {(latest.progress * 100).toFixed(0)}%
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
