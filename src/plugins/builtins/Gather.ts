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
    pytorch(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      const idx = inputVars[1] ?? "index";
      return { attr: null, forward: `torch.gather(${x}, ${axis}, ${idx})` };
    },
    keras(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      const idx = inputVars[1] ?? "indices";
      return { attr: null, forward: `tf.gather(${x}, ${idx}, axis=${axis})` };
    },
    candle(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      const idx = inputVars[1] ?? "index";
      return { attr: null, forward: `${x}.gather(&${idx}, ${axis})?` };
    },
  },
};

export default Gather;
