declare module "@mkkellogg/gaussian-splats-3d" {
  import type { Group } from "three";

  export class DropInViewer extends Group {
    constructor(options?: {
      sharedMemoryForWorkers?: boolean;
      showLoadingUI?: boolean;
      gpuAcceleratedSort?: boolean;
    });
    addSplatScene(
      path: string,
      options?: {
        splatAlphaRemovalThreshold?: number;
        showLoadingUI?: boolean;
        position?: number[];
        rotation?: number[];
        scale?: number[];
      },
    ): Promise<void>;
    dispose(): Promise<void>;
  }
}
