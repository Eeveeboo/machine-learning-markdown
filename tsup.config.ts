import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "bin/mlmd.ts"],
  format: ["esm"],
  dts: true,
  clean: true,
});
