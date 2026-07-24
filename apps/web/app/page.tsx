import Link from "next/link";
import { listVideos, listJobsForVideo } from "@/lib/store";

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
                    {video.meta.duration_s != null
                      ? ` · ${video.meta.duration_s.toFixed(1)}s`
                      : ""}
                    {latest
                      ? ` · ${latest.stage} ${(latest.progress * 100).toFixed(0)}%`
                      : ""}
                  </div>
                </Link>
              );
            }),
          )}
        </div>
      )}
    </div>
  );
}
