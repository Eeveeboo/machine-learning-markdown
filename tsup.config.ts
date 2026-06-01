import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "bin/mlmd.ts"],
  format: ["esm"],
  platform: "node",
  external: ["vscode-languageserver", "vscode-languageserver-textdocument"],
  dts: true,
  clean: true,
});
