import type { BlockPlugin } from "../types.js";

const Repeat: BlockPlugin = {
  inputs: ["x"],
  outputs: ["y"],

  params: [{ name: "times", type: "number", required: true }],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Repeat requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const tv = block.params["times"];
      const times = tv && tv.kind === "number" ? tv.value : 1;
      return { forward: `${outputVars[0]} = ${inputVars[0]}.repeat(${times})` };
    },
    keras(block, inputVars, outputVars) {
      const tv = block.params["times"];
      const times = tv && tv.kind === "number" ? tv.value : 1;
      return { forward: `${outputVars[0]} = keras.layers.RepeatVector(${times})(${inputVars[0]})` };
    },
    candle(block, inputVars, outputVars) {
      const tv = block.params["times"];
      const times = tv && tv.kind === "number" ? tv.value : 1;
      return { forward: `${outputVars[0]} = ${inputVars[0]}.repeat(&[${times}])?` };
    },
  },
};

export default Repeat;
