"use client";

import { useEffect, useState } from "react";
import type { Group } from "three";

/**
 * Drop-in Gaussian splat (.ply / .splat / .ksplat) for R3F / Three scenes.
 * Uses @mkkellogg/gaussian-splats-3d with workers (sharedMemory disabled for Next).
 */
export function GaussianSplat({
  url,
  visible = true,
}: {
  url: string;
  visible?: boolean;
}) {
  const [viewer, setViewer] = useState<Group | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    let dropIn: { dispose?: () => Promise<void> } | null = null;

    void (async () => {
      try {
        const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
        if (disposed) return;
        const v = new GaussianSplats3D.DropInViewer({
          sharedMemoryForWorkers: false,
          showLoadingUI: false,
          gpuAcceleratedSort: false,
        });
        dropIn = v;
        await v.addSplatScene(url, {
          splatAlphaRemovalThreshold: 5,
          showLoadingUI: false,
        });
        if (disposed) {
          await v.dispose?.();
          return;
        }
        setViewer(v as unknown as Group);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.error("[GaussianSplat]", msg);
        if (!disposed) setError(msg);
      }
    })();

    return () => {
      disposed = true;
      setViewer(null);
      void dropIn?.dispose?.().catch(() => undefined);
    };
  }, [url]);

  if (error || !viewer || !visible) return null;
  return <primitive object={viewer} visible={visible} />;
}
