import Link from "next/link";
import { notFound } from "next/navigation";
import {
  getBallTracks,
  getCalibration,
  getCourt3d,
  getPlayersTracks,
  getVideo,
  listJobsForVideo,
} from "@/lib/store";
import { VideoWorkspace } from "@/components/VideoWorkspace";
import type { Court3dFile } from "@/components/Court3D";

export const dynamic = "force-dynamic";

export default async function VideoDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const video = await getVideo(id);
  if (!video) notFound();

  const [jobs, calibration, tracks, ball, court3d] = await Promise.all([
    listJobsForVideo(id),
    getCalibration(id),
    getPlayersTracks(id),
    getBallTracks(id),
    getCourt3d(id),
  ]);

  return (
    <div className="stack">
      <div>
        <Link href="/" className="back-link">
          ← Library
        </Link>
        <div className="page-header">
          <h1>{video.name}</h1>
          <p className="lede">
            {video.original_filename ?? video.id}
            {video.meta.width && video.meta.height
              ? ` · ${video.meta.width}×${video.meta.height}`
              : ""}
            {video.meta.duration_s != null
              ? ` · ${video.meta.duration_s.toFixed(1)}s`
              : ""}
          </p>
        </div>
      </div>

      <VideoWorkspace
        video={video}
        initialJobs={jobs}
        initialCalibration={calibration}
        initialTracks={tracks}
        initialBall={ball}
        initialCourt3d={(court3d as Court3dFile | null) ?? null}
      />
    </div>
  );
}
