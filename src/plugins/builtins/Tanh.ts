import type { BlockPlugin } from "../types.js";

const Tanh: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Tanh requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars) {
      return {
        attr: { name: block.id, init: "nn.Tanh()" },
        forward: `self.${block.id}(${inputVars[0] ?? "x"})`,
      };
    },
    keras(_block, inputVars) {
      return {
        attr: null,
        forward: `keras.layers.Activation('tanh')(${inputVars[0] ?? "x"})`,
      };
    },
    candle(_block, inputVars) {
      const x = inputVars[0] ?? "x";
      return { attr: null, forward: `${x}.tanh()?` };
    },
  },
};

export default Tanh;
