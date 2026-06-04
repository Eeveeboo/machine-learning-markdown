import type { BlockPlugin } from "../types.js";

const MatMul: BlockPlugin = {
  inputs: ["a", "b"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length < 2) throw new Error("MatMul requires two inputs");
    const a = inputs[0];
    const b = inputs[1];
    const lastA = a[a.length - 1];
    const lastB = b[b.length - 1];
    const secondLastB = b.length >= 2 ? b[b.length - 2] : 0;
    // Same transpose detection as codegen — matching last dims with incompatible inner dims
    const needsTranspose = lastA === lastB && lastA !== secondLastB;
    // Determine N dimension: with transpose, use b's second-to-last dim (inner dim), else use b's last dim
    const N = needsTranspose ? (b.length >= 2 ? b[b.length - 2] : 0) : b[b.length - 1];
    if (a.length === 2 && b.length === 2) {
      return [[a[0], N]];
    }
    // Batched: [..., M, K] * [..., K, N] = [..., M, N]
    const result = a.slice(0, -2);
    if (a.length >= 2) result.push(a[a.length - 2]);  // M
    result.push(N);
    return [result];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = torch.matmul(${inputVars[0]}, ${inputVars[1]})` };
    },
    keras(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = keras.layers.Dot(axes=-1)([${inputVars.join(", ")}])` };
    },
    candle(block, inputVars, outputVars) {
      // Detect if we need to transpose the second input for Q*K^T style attention.
      // This occurs when both inputs have the same shape and their last dims match
      // but standard matmul would fail (last dim of A ≠ second-to-last of B).
      const shapeA = block.inputShapes?.[0];
      const shapeB = block.inputShapes?.[1];
      const needsTranspose =
        shapeA && shapeB &&
        shapeA.length >= 2 && shapeB.length >= 2 &&
        shapeA[shapeA.length - 1] === shapeB[shapeB.length - 1] &&
        shapeA[shapeA.length - 1] !== shapeB[shapeB.length - 2];
      const rhs = needsTranspose ? `${inputVars[1]}.t()?` : `${inputVars[1]}`;
      return { forward: `${outputVars[0]} = ${inputVars[0]}.matmul(&${rhs})?;` };
    },
  },
};

export default MatMul;
