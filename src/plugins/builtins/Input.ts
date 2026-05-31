import type { Block } from "../../ast/graph.js";
import type { BlockPlugin, BlockCodegenResult } from "../types.js";
import { requireShape } from "./_helpers.js";

const Input: BlockPlugin = {
  inputs: [],
  outputs: ["output"],

  params: [{ name: "shape", type: "shape", required: true }],
  inferShape(_inputs, params) {
    return [requireShape(params, "shape")];
  },

  paramCount: () => 0,

  codegen: {
    pytorch(block: Block, inputVars: string[]): BlockCodegenResult {
      const mainIn = inputVars[0] ?? "x";
      return { attr: null, forward: mainIn };
    },

    keras(_block: Block, _inputVars: string[]): BlockCodegenResult {
      // Keras Input is handled specially in the generator; no op here
      return { attr: null, forward: "" };
    },

    candle(_block: Block, _inputVars: string[]): BlockCodegenResult {
      // Input in candle is the initial tensor; no op
      return { attr: null, forward: "" };
    },
  },
};

export default Input;
