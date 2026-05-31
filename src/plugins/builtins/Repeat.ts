import type { BlockPlugin } from "../types.js";

const Repeat: BlockPlugin = {
  inputs: ["x"],
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Repeat requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const tv = block.params["times"];
      const times = tv && tv.kind === "number" ? tv.value : 1;
      return { attr: null, forward: `${x}.repeat(${times})` };
    },
    keras(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const tv = block.params["times"];
      const times = tv && tv.kind === "number" ? tv.value : 1;
      return { attr: null, forward: `keras.layers.RepeatVector(${times})(${x})` };
    },
    candle(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const tv = block.params["times"];
      const times = tv && tv.kind === "number" ? tv.value : 1;
      return { attr: null, forward: `${x}.repeat(&[${times}])?` };
    },
  },
};

export default Repeat;
