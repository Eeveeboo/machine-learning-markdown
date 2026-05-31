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
    pytorch(_block, inputVars) {
      const a = inputVars[0] ?? "x";
      const b = inputVars[1] ?? "x";
      return { attr: null, forward: `torch.matmul(${a}, ${b})` };
    },
    keras(_block, inputVars) {
      return { attr: null, forward: `keras.layers.Dot(axes=-1)([${inputVars.join(", ")}])` };
    },
    candle(_block, inputVars) {
      const a = inputVars[0] ?? "x";
      const b = inputVars[1] ?? "x";
      return { attr: null, forward: `${a}.matmul(&${b})?` };
    },
  },
};

export default MatMul;
