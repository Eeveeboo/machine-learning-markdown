import type { Block } from "../../ast/graph.js";
import type { BlockPlugin, BlockCodegenResult } from "../types.js";

const Output: BlockPlugin = {
  inputs: ["input"],
  outputs: [],

  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Output requires an input");
    return [inputs[0]];
  },

  paramCount: () => 0,

  codegen: {
    pytorch(_block: Block, inputVars: string[]): BlockCodegenResult {
      const mainIn = inputVars[0] ?? "x";
      return { attr: null, forward: `return ${mainIn}` };
    },

    keras(_block: Block, inputVars: string[]): BlockCodegenResult {
      // Keras Output is handled specially in the generator
      return { attr: null, forward: "" };
    },

    candle(_block: Block, inputVars: string[]): BlockCodegenResult {
      const mainIn = inputVars[0] ?? "x";
      return { attr: null, forward: `Ok(${mainIn})` };
    },
  },
};

export default Output;
