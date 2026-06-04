import type { BlockPlugin } from "../types.js";

const Split: BlockPlugin = {
  inputs: ["x"],
  outputs: ["y"],

  params: [{ name: "chunks", type: "number", required: true }, { name: "axis", type: "number", required: false }],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("Split requires an input");
    const nv = params["chunks"];
    const N = nv && nv.kind === "number" ? nv.value : NaN;
    const av = params["axis"];
    const axis = av && av.kind === "number" ? av.value : 0;
    const outShape = [...inputs[0]];
    outShape[axis] = Math.floor(inputs[0][axis] / N);
    return Array.from({ length: N }, () => [...outShape]);
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const nv = block.params["chunks"];
      const N = nv && nv.kind === "number" ? nv.value : 2;
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      return { forward: `${outputVars[0]} = torch.chunk(${inputVars[0]}, ${N}, dim=${axis})` };
    },
    keras(block, inputVars, outputVars) {
      const nv = block.params["chunks"];
      const N = nv && nv.kind === "number" ? nv.value : 2;
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      return { forward: `${outputVars[0]} = tf.split(${inputVars[0]}, ${N}, axis=${axis})` };
    },
    candle(block, inputVars, outputVars) {
      const nv = block.params["chunks"];
      const N = nv && nv.kind === "number" ? nv.value : 2;
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      return { forward: `${outputVars[0]} = ${inputVars[0]}.chunk(${N}, ${axis})?;` };
    },
  },
};

export default Split;
