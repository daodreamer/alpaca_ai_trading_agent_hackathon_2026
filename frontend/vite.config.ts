import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The dashboard is served by FastAPI in production, so the build lands inside
// the Python package rather than in a sibling `dist/`. One process serves the
// page and the API, which is one fewer thing to start on demo day and removes
// the CORS question entirely.
//
// `dev` proxies the API to the same backend, so `npm run dev` behaves like the
// built page with hot reload on top.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: path.resolve(__dirname, "../backend/src/alphagate/interface/static"),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
})
