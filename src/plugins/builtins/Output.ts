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
    pytorch(_block: Block, inputVars: string[], _outputVars: string[]): BlockCodegenResult {
      return { forward: `return ${inputVars[0]}` };
    },

    keras(_block: Block, inputVars: string[], _outputVars: string[]): BlockCodegenResult {
      return { forward: `return ${inputVars[0]}` };
    },

    candle(_block: Block, inputVars: string[], _outputVars: string[]): BlockCodegenResult {
      return { forward: `Ok(${inputVars[0]})` };
    },
  },
};

export default Output;
