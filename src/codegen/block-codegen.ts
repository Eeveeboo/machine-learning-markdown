export type { BlockCodegenResult, BlockCodegenFn } from "../plugins/types.js";
import type { BlockCodegenFn, BlockCodegenResult } from "../plugins/types.js";
import type { Block } from "../ast/graph.js";

export const codegenRegistry = new Map<string, Record<string, BlockCodegenFn>>();

export function registerBlockCodegen(
  blockType: string,
  target: string,
  fn: BlockCodegenFn,
): void {
  let targetMap = codegenRegistry.get(blockType);
  if (!targetMap) {
    targetMap = {};
    codegenRegistry.set(blockType, targetMap);
  }
  targetMap[target] = fn;
}

export function getBlockCodegen(
  blockType: string,
  target: string,
): BlockCodegenFn | undefined {
  return codegenRegistry.get(blockType)?.[target];
}

export function fallbackCodegen(block: Block, _inputVars: string[]): BlockCodegenResult {
  return {
    attr: null,
    forward: `# TODO: codegen not implemented for ${block.type} (target: unknown)`,
  };
}

export function getBlockCodegenWithFallback(
  blockType: string,
  target: string,
): BlockCodegenFn {
  return getBlockCodegen(blockType, target) ?? fallbackCodegenFor(target);
}

export function fallbackCodegenFor(target: string): BlockCodegenFn {
  return (block: Block, inputVars: string[]): BlockCodegenResult => {
    const mainIn = inputVars[0] ?? "x";
    const comment = `/* ${block.type} — custom block, passthrough in generated code */`;
    switch (target) {
      case "candle":
      case "pytorch":
        return { attr: null, forward: `${mainIn} ${comment}` };
      case "keras":
        return { attr: null, forward: mainIn };
      default:
        return { attr: null, forward: `${mainIn} ${comment}` };
    }
  };
}
