/**
 * Plugin loader — discovers and imports BlockPlugin files from a directory.
 *
 * ## Resolution order
 * For each unique base-name found in the directory the loader tries:
 *   1. `<name>.mjs`
 *   2. `<name>.js`
 *   3. `<name>.ts`  (requires `tsx` or compatible runtime; see types.ts)
 *
 * If a plugin exports `inferShape`, it is also registered in the block
 * registry via `registerBlock` so the rest of the pipeline can use it.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { BlockPlugin } from "./types.js";
import { registerBlock } from "../blocks/registry.js";

const EXTENSIONS = [".mjs", ".js", ".ts"] as const;

function isBlockPlugin(value: unknown): value is BlockPlugin {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return Array.isArray(v["inputs"]) && Array.isArray(v["outputs"]);
}

/**
 * Scan `dir` for plugin files and dynamically import each one.
 *
 * Returns a Map from block-type-name → BlockPlugin. Plugins that provide
 * `inferShape` are additionally registered in the global block registry.
 */
export async function loadPlugins(dir: string): Promise<Map<string, BlockPlugin>> {
  const result = new Map<string, BlockPlugin>();

  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return result;
  }

  // Collect unique base names, preserving first-seen extension priority
  const candidates = new Map<string, string>(); // baseName → full resolved path

  for (const ext of EXTENSIONS) {
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      if (!entry.name.endsWith(ext)) continue;
      const baseName = entry.name.slice(0, -ext.length);
      if (!candidates.has(baseName)) {
        candidates.set(baseName, path.resolve(dir, entry.name));
      }
    }
  }

  for (const [blockType, filePath] of candidates) {
    let mod: Record<string, unknown>;
    try {
      mod = await import(filePath) as Record<string, unknown>;
    } catch (err) {
      console.warn(`[plugin-loader] Failed to import ${filePath}: ${(err as Error).message}`);
      continue;
    }

    // Prefer named export matching block type, then fall back to default
    const candidate = (blockType in mod && mod[blockType] !== undefined)
      ? mod[blockType]
      : mod["default"];

    if (!isBlockPlugin(candidate)) {
      console.warn(`[plugin-loader] ${filePath} does not export a valid BlockPlugin — skipping`);
      continue;
    }

    result.set(blockType, candidate);

    if (typeof candidate.inferShape === "function") {
      registerBlock({
        name: blockType,
        params: [],
        inferShape: candidate.inferShape.bind(candidate),
        ...(candidate.paramCount !== undefined
          ? { paramCount: candidate.paramCount.bind(candidate) }
          : {}),
      });
    }
  }

  return result;
}
