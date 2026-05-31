import { readFileSync } from "node:fs";
import { resolve } from "node:path";
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

  return validateConfig(parsed);
}
