import type { BlockPlugin } from "../types.js";

const SiLU: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("SiLU requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars) {
      return {
        attr: { name: block.id, init: "nn.SiLU()" },
        forward: `self.${block.id}(${inputVars[0] ?? "x"})`,
      };
    },
    keras(_block, inputVars) {
      return {
        attr: null,
        forward: `keras.layers.Activation('swish')(${inputVars[0] ?? "x"})`,
      };
    },
    candle(_block, inputVars) {
      const x = inputVars[0] ?? "x";
      return { attr: null, forward: `${x}.silu()?` };
    },
  },
};

export default SiLU;
