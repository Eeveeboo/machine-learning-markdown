import type { Block, Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";
import type { ParamSpec } from "../blocks/types.js";
import type { RenderContext } from "../visualize/render-context.js";

export interface CandleInit {
  /** Struct field declaration for statically-typed targets. */
  field: string;
  /** Init body statement for statically-typed targets. */
  body: string;
}

export interface BlockCodegenResult {
  /**
   * Optional init-time declaration.
   * - `null` for stateless blocks (activations, merges, arithmetic).
   * - A `string` for most targets (PyTorch, Keras, etc.).
   * - A `CandleInit` for Candle (Rust) backends.
   */
  init?: string | CandleInit | null;
  forward: string;
}

export type BlockCodegenFn = (block: Block, inputVars: string[], outputVars: string[]) => BlockCodegenResult;

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
  params?: ParamSpec[];
  render?(ctx: RenderContext): void;
  codegen?: Record<string, BlockCodegenFn>;
  inferShape?(inputs: Shape[], params: Record<string, ParamValue>): Shape[];
  paramCount?(inputs: Shape[], params: Record<string, ParamValue>): number;
  showDepth?: boolean;
}
