import type { BlockPlugin } from "../types.js";

const Split: BlockPlugin = {
  inputs: ["x"],
  outputs: ["y"],
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
    pytorch(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const nv = block.params["chunks"];
      const N = nv && nv.kind === "number" ? nv.value : 2;
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      return { attr: null, forward: `torch.chunk(${x}, ${N}, dim=${axis})` };
    },
    keras(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const nv = block.params["chunks"];
      const N = nv && nv.kind === "number" ? nv.value : 2;
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      return { attr: null, forward: `tf.split(${x}, ${N}, axis=${axis})` };
    },
    candle(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const nv = block.params["chunks"];
      const N = nv && nv.kind === "number" ? nv.value : 2;
      const av = block.params["axis"];
      const axis = av && av.kind === "number" ? av.value : 0;
      return { attr: null, forward: `${x}.chunk(${N}, ${axis})?` };
    },
  },
};

export default Split;
