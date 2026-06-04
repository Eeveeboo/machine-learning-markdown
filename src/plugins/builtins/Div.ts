import type { BlockPlugin } from "../types.js";

const Div: BlockPlugin = {
  inputs: ["a", "b"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Div requires inputs");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = torch.div(${inputVars[0]}, ${inputVars[1]})` };
    },
    keras(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = keras.layers.Divide()([${inputVars.join(", ")}])` };
    },
    candle(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = (&${inputVars[0]} / &${inputVars[1]})?` };
    },
  },
};

export default Div;
