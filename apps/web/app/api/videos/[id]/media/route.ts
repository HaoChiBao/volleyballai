import { createReadStream, promises as fs } from "fs";
import type { ReadStream } from "fs";
import { NextResponse } from "next/server";
import { filePathForVideoAsset, getVideo } from "@/lib/store";

export const runtime = "nodejs";

const KIND = new Set(["source", "work", "thumb"] as const);
type Kind = "source" | "work" | "thumb";

/**
 * Convert a Node fs ReadStream to a Web ReadableStream without using
 * Readable.toWeb(). Seeking HTML5 video cancels in-flight range requests;
 * toWeb() often races and throws ERR_INVALID_STATE ("Controller is already closed").
 */
function toWebStream(
  nodeStream: ReadStream,
  signal?: AbortSignal | null,
): ReadableStream<Uint8Array> {
  let closed = false;

  const destroyNode = () => {
    try {
      nodeStream.destroy();
    } catch {
      // ignore
    }
  };

  const safeClose = (controller: ReadableStreamDefaultController<Uint8Array>) => {
    if (closed) {
      destroyNode();
      return;
    }
    closed = true;
    destroyNode();
    try {
      controller.close();
    } catch {
      // Already closed by cancel() / prior end.
    }
  };

  const safeError = (
    controller: ReadableStreamDefaultController<Uint8Array>,
    err: unknown,
  ) => {
    if (closed) {
      destroyNode();
      return;
    }
    closed = true;
    destroyNode();
    try {
      controller.error(err);
    } catch {
      // Already closed.
    }
  };

  return new ReadableStream<Uint8Array>({
    start(controller) {
      const onAbort = () => safeClose(controller);
      if (signal) {
        if (signal.aborted) {
          onAbort();
          return;
        }
        signal.addEventListener("abort", onAbort, { once: true });
        nodeStream.once("close", () => {
          signal.removeEventListener("abort", onAbort);
        });
      }

      nodeStream.on("data", (chunk: string | Buffer) => {
        if (closed) return;
        const bytes =
          typeof chunk === "string" ? Buffer.from(chunk) : chunk;
        try {
          controller.enqueue(new Uint8Array(bytes));
        } catch {
          // Consumer cancelled mid-chunk (video seek).
          closed = true;
          destroyNode();
        }
      });
      nodeStream.once("end", () => safeClose(controller));
      nodeStream.once("error", (err) => {
        // Abort/destroy after cancel is expected — don't surface as stream error.
        if (closed) return;
        if (
          err &&
          typeof err === "object" &&
          "code" in err &&
          (err as { code?: string }).code === "ERR_STREAM_PREMATURE_CLOSE"
        ) {
          safeClose(controller);
          return;
        }
        safeError(controller, err);
      });
    },
    cancel() {
      closed = true;
      destroyNode();
    },
  });
}

function parseByteRange(
  header: string,
  fileSize: number,
): { start: number; end: number } | "invalid" {
  const match = /^bytes=(\d*)-(\d*)$/i.exec(header.trim());
  if (!match) return "invalid";

  const startTok = match[1];
  const endTok = match[2];

  // Suffix form: bytes=-N (last N bytes)
  if (!startTok && endTok) {
    const suffix = Number(endTok);
    if (!Number.isFinite(suffix) || suffix <= 0) return "invalid";
    const start = Math.max(0, fileSize - suffix);
    return { start, end: fileSize - 1 };
  }

  const start = startTok ? Number(startTok) : 0;
  if (!Number.isFinite(start) || start < 0) return "invalid";

  const end = endTok ? Number(endTok) : fileSize - 1;
  if (!Number.isFinite(end) || end < start) return "invalid";

  if (start >= fileSize) return "invalid";
  return { start, end: Math.min(end, fileSize - 1) };
}

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const video = await getVideo(id);
  if (!video) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const url = new URL(request.url);
  const kind = (url.searchParams.get("kind") ?? "work") as Kind;
  if (!KIND.has(kind)) {
    return NextResponse.json({ error: "Invalid kind" }, { status: 400 });
  }

  const filePath = filePathForVideoAsset(id, kind);
  let fileSize: number;
  try {
    fileSize = (await fs.stat(filePath)).size;
  } catch {
    return NextResponse.json({ error: "File missing" }, { status: 404 });
  }

  const contentType = kind === "thumb" ? "image/jpeg" : "video/mp4";

  // Images don't need range seeking — keep a simple full response.
  if (kind === "thumb") {
    try {
      const data = await fs.readFile(filePath);
      return new NextResponse(data, {
        headers: {
          "Content-Type": contentType,
          "Content-Length": String(fileSize),
          "Cache-Control": "no-store",
        },
      });
    } catch {
      return NextResponse.json({ error: "File missing" }, { status: 404 });
    }
  }

  // HTML5 video scrubbing requires byte-range (206) responses.
  const rangeHeader = request.headers.get("range");
  if (rangeHeader) {
    const parsed = parseByteRange(rangeHeader, fileSize);
    if (parsed === "invalid") {
      return new NextResponse(null, {
        status: 416,
        headers: {
          "Content-Range": `bytes */${fileSize}`,
          "Accept-Ranges": "bytes",
        },
      });
    }

    const { start, end } = parsed;
    const chunkSize = end - start + 1;
    const stream = createReadStream(filePath, { start, end });

    return new NextResponse(toWebStream(stream, request.signal), {
      status: 206,
      headers: {
        "Content-Type": contentType,
        "Content-Length": String(chunkSize),
        "Content-Range": `bytes ${start}-${end}/${fileSize}`,
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
      },
    });
  }

  // Full file — still advertise Accept-Ranges so the browser enables seeking.
  const stream = createReadStream(filePath);
  return new NextResponse(toWebStream(stream, request.signal), {
    status: 200,
    headers: {
      "Content-Type": contentType,
      "Content-Length": String(fileSize),
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-store",
    },
  });
}
