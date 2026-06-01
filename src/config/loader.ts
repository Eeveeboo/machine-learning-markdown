import { readFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import type { MlmdConfig } from "./schema.js";
import { validateConfig } from "./schema.js";

export function loadConfig(cwd?: string): MlmdConfig | null {
  const dir = cwd ?? process.cwd();

  let filePath: string;
  if (process.env["MLMD_CONFIG"]) {
    filePath = resolve(process.env["MLMD_CONFIG"]);
  } else {
    filePath = resolve(dir, ".mlmdrc");
  }

  let raw: string;
  try {
    raw = readFileSync(filePath, "utf-8");
  } catch {
    return null;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    console.warn(`[mlmd] config: invalid JSON in ${filePath}`);
    return null;
  }

  const config = validateConfig(parsed);
  if (!config) return null;

  const resolvedCwd = resolve(dir);

  if (config.plugins) {
    const resolvedPlugins = resolve(dir, config.plugins);
    if (!resolvedPlugins.startsWith(resolvedCwd + sep) && resolvedPlugins !== resolvedCwd) {
      console.warn(`[mlmd] config: plugins path must be within project root: ${config.plugins}`);
      return null;
    }
  }

  if (config.targets) {
    for (const target of config.targets) {
      const resolvedOut = resolve(dir, target.out);
      if (!resolvedOut.startsWith(resolvedCwd + sep) && resolvedOut !== resolvedCwd) {
        console.warn(`[mlmd] config: target.out must be within project root: ${target.out}`);
        return null;
      }
    }
  }

  return config;
}
