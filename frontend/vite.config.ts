import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");
  const allowedHosts = (env.VITE_ALLOWED_HOSTS ?? "")
    .split(",")
    .map((host) => host.trim())
    .filter(Boolean);
  const apiProxyTarget = env.API_PROXY_TARGET?.trim() || "http://127.0.0.1:8000";
  const apiProxy = {
    "/api": {
      target: apiProxyTarget,
      changeOrigin: true,
      rewrite: (path: string) => path.replace(/^\/api/, ""),
    },
  };

  return {
    envDir: "..",
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      strictPort: true,
      allowedHosts,
      proxy: apiProxy,
    },
    preview: {
      host: "127.0.0.1",
      port: 4173,
      strictPort: true,
      allowedHosts,
      proxy: apiProxy,
    },
  };
});
