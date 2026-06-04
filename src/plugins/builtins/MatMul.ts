import type { BlockPlugin } from "../types.js";

const MatMul: BlockPlugin = {
  inputs: ["a", "b"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length < 2) throw new Error("MatMul requires two inputs");
    const [A] = inputs[0];
    const [, C] = inputs[1];
    return [[A, C]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = torch.matmul(${inputVars[0]}, ${inputVars[1]})` };
    },
    keras(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = keras.layers.Dot(axes=-1)([${inputVars.join(", ")}])` };
    },
    candle(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = ${inputVars[0]}.matmul(&${inputVars[1]})?` };
    },
  },
};

export default MatMul;
