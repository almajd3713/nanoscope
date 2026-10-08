import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  // The built app ships inside the Python package (wheel and image); make web builds it.
  build: { outDir: "../nanoscope/server/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  test: { environment: "jsdom", setupFiles: ["test/setup.ts"] },
});
