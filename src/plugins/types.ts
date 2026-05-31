import type { Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";
import type { RenderContext, CodegenContext } from "../visualize/render-context.js";

/**
 * Interface that every plugin must satisfy.
 *
 * ## TS compilation note
 * Node cannot import `.ts` files directly at runtime. The loader resolves
 * plugin files in this order of preference:
 *   1. `<name>.mjs`  — native ESM, works out of the box
 *   2. `<name>.js`   — compiled JS, works out of the box
 *   3. `<name>.ts`   — TypeScript source; only usable when `tsx` (or a similar
 *                      ts-capable runtime) is present on PATH. Without `tsx`
 *                      the import will fail with a "Unknown file extension"
 *                      error. Ship plugins as `.mjs` or pre-compiled `.js`
 *                      for production use.
 *
 * Convention: the filename without extension becomes the block type name.
 * e.g. `MyCustomLayer.mjs` registers the block type `"MyCustomLayer"`.
 *
 * A plugin file must export either:
 *   - a **default** export that is a `BlockPlugin` object, or
 *   - a **named** export whose key matches the filename (block type name).
 */
export interface BlockPlugin {
  inputs: string[];
  outputs: string[];
  render?(ctx: RenderContext): void;
  codegen?(ctx: CodegenContext): string;
  inferShape?(inputs: Shape[], params: Record<string, ParamValue>): Shape[];
  paramCount?(inputs: Shape[], params: Record<string, ParamValue>): number;
}
