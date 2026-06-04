import type { BlockPlugin } from "../types.js";

const Concat: BlockPlugin = {
  inputs: ["a", "b"],
  showDepth: false,
  outputs: ["y"],

  params: [{ name: "axis", type: "number", required: false }],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("Concat requires inputs");
    const v = params["axis"];
    const axis = v && v.kind === "number" ? v.value : 1;
    const out = [...inputs[0]];
    for (let i = 1; i < inputs.length; i++) {
      out[axis] = out[axis] + inputs[i][axis];
    }
    return [out];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const v = block.params["axis"];
      const dim = v && v.kind === "number" ? v.value : 1;
      return { forward: `${outputVars[0]} = torch.cat([${inputVars.join(", ")}], dim=${dim})` };
    },
    keras(block, inputVars, outputVars) {
      const v = block.params["axis"];
      const axis = v && v.kind === "number" ? v.value : -1;
      return { forward: `${outputVars[0]} = keras.layers.Concatenate(axis=${axis})([${inputVars.join(", ")}])` };
    },
    candle(_block, inputVars, outputVars) {
      const tensors = inputVars.map((v) => `&${v}`).join(", ");
      return { forward: `${outputVars[0]} = Tensor::cat(&[${tensors}], 1)?` };
    },
  },
};

export default Concat;
