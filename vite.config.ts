import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: "electron/renderer",
  plugins: [react()],
  build: {
    outDir: "../../dist/renderer",
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: 17888,
    strictPort: true,
  },
});
