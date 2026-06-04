import type { BlockPlugin } from "../types.js";

const Gather: BlockPlugin = {
  inputs: ["x"],
  outputs: ["y"],

  params: [{ name: "axis", type: "number", required: false }],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Gather requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      const idx = inputVars[1] ?? "index";
      return { forward: `${outputVars[0]} = torch.gather(${inputVars[0]}, ${axis}, ${idx})` };
    },
    keras(block, inputVars, outputVars) {
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      const idx = inputVars[1] ?? "indices";
      return { forward: `${outputVars[0]} = tf.gather(${inputVars[0]}, ${idx}, axis=${axis})` };
    },
    candle(block, inputVars, outputVars) {
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      const idx = inputVars[1] ?? "index";
      return { forward: `${outputVars[0]} = ${inputVars[0]}.gather(&${idx}, ${axis})?` };
    },
  },
};

export default Gather;
