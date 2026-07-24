"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a video file first.");
      return;
    }
    setBusy(true);
    setError(null);
    setProgress(0);

    try {
      const form = new FormData();
      form.append("file", file);

      const videoId = await new Promise<string>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/videos");
        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable) {
            setProgress(ev.loaded / ev.total);
          }
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const data = JSON.parse(xhr.responseText) as { id: string };
              resolve(data.id);
            } catch {
              reject(new Error("Invalid response from server"));
            }
          } else {
            reject(
              new Error(xhr.responseText || `Upload failed (${xhr.status})`),
            );
          }
        };
        xhr.onerror = () => reject(new Error("Network error during upload"));
        xhr.send(form);
      });

      const jobRes = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId }),
      });
      if (!jobRes.ok) {
        const text = await jobRes.text();
        throw new Error(text || "Failed to create job");
      }

      router.push(`/videos/${videoId}`);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <div className="page-header">
        <h1>Upload</h1>
        <p className="lede">
          Add a short volleyball clip. An analysis job starts automatically after
          upload.
        </p>
      </div>

      <form className="card stack" onSubmit={onSubmit}>
        <div className="file-field">
          <label htmlFor="video-file">Video file</label>
          <input
            id="video-file"
            type="file"
            accept="video/mp4,video/*"
            disabled={busy}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>
        {file ? (
          <p className="hint">Selected: {file.name}</p>
        ) : null}
        {busy ? (
          <div className="stack">
            <div className="meta-line">
              Uploading… {(progress * 100).toFixed(0)}%
            </div>
            <div className="progress">
              <span style={{ width: `${Math.max(progress * 100, 2)}%` }} />
            </div>
          </div>
        ) : null}
        {error ? <div className="error">{error}</div> : null}
        <div className="row">
          <button type="submit" disabled={busy || !file}>
            {busy ? "Working…" : "Upload and analyze"}
          </button>
        </div>
      </form>
    </div>
  );
}
