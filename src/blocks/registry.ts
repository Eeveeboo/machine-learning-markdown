import type { BlockDef } from "./types.js";

export const registry: Map<string, BlockDef> = new Map();

/**
 * Register a block definition. Duplicate names silently overwrite the previous entry.
 */
export function registerBlock(def: BlockDef): void {
  registry.set(def.name, def);
}

export function lookupBlock(name: string): BlockDef | undefined {
  return registry.get(name);
}
