import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Allow serving uploaded media from the local .data directory via API routes.
  experimental: {
    serverActions: {
      bodySizeLimit: "512mb",
    },
  },
  eslint: {
    // Flat ESLint + next/core-web-vitals currently warns during build; keep typecheck.
    ignoreDuringBuilds: true,
  },
  transpilePackages: ["@volleyballai/types", "@volleyballai/court-math"],
  outputFileTracingRoot: path.join(__dirname, "../.."),
};

export default nextConfig;
