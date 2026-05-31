import type { Graph, Block } from "../../ast/graph.js";
import type { BlockDef } from "../../blocks/types.js";
import { registerTarget } from "../target.js";
import type { GeneratedFile } from "../target.js";
import { topoSort, buildAdjacency } from "../../ast/graph-utils.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getNum(block: Block, key: string, fallback?: number): number | undefined {
  const p = block.params[key];
  if (p && p.kind === "number") return p.value;
  return fallback;
}

function getStr(block: Block, key: string, fallback?: string): string | undefined {
  const p = block.params[key];
  if (p && p.kind === "string") return p.value;
  if (p && p.kind === "bareword") return p.value;
  return fallback;
}

function shapeComment(shapes: number[][]): string {
  if (!shapes.length) return "";
  return "  # " + shapes.map((s) => `[${s.join(", ")}]`).join(", ");
}

// ---------------------------------------------------------------------------
// Layer expression generation (Keras functional API)
// ---------------------------------------------------------------------------

/** Returns the Keras layer expression for a block applied to its input(s). */
function blockToLayerExpr(block: Block, inputVars: string[]): string | null {
  const t = block.type;
  const mainIn = inputVars[0] ?? "x";

  switch (t) {
    case "Input":
    case "Output":
      return null;

    case "Linear": {
      const outF = getNum(block, "out_features") ?? getNum(block, "units") ?? 0;
      return `keras.layers.Dense(${outF})(${mainIn})`;
    }

    case "Conv2d": {
      const filters = getNum(block, "filters") ?? getNum(block, "out_channels") ?? 0;
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 3;
      const stride = getNum(block, "stride") ?? 1;
      const padding = getStr(block, "padding") ?? (getNum(block, "padding") === 0 ? "valid" : "same");
      return `keras.layers.Conv2D(${filters}, ${kernel}, strides=${stride}, padding='${padding}')(${mainIn})`;
    }

    case "Conv1d": {
      const filters = getNum(block, "filters") ?? getNum(block, "out_channels") ?? 0;
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 3;
      const stride = getNum(block, "stride") ?? 1;
      const padding = getStr(block, "padding") ?? "valid";
      return `keras.layers.Conv1D(${filters}, ${kernel}, strides=${stride}, padding='${padding}')(${mainIn})`;
    }

    case "Conv3d": {
      const filters = getNum(block, "filters") ?? getNum(block, "out_channels") ?? 0;
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 3;
      const stride = getNum(block, "stride") ?? 1;
      const padding = getStr(block, "padding") ?? "valid";
      return `keras.layers.Conv3D(${filters}, ${kernel}, strides=${stride}, padding='${padding}')(${mainIn})`;
    }

    case "TransposedConv2d": {
      const filters = getNum(block, "filters") ?? getNum(block, "out_channels") ?? 0;
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 3;
      const stride = getNum(block, "stride") ?? 1;
      const padding = getStr(block, "padding") ?? "valid";
      return `keras.layers.Conv2DTranspose(${filters}, ${kernel}, strides=${stride}, padding='${padding}')(${mainIn})`;
    }

    case "Embedding": {
      const vocab = getNum(block, "vocab_size") ?? 0;
      const embed = getNum(block, "embed_dim") ?? getNum(block, "embedding_dim") ?? 0;
      return `keras.layers.Embedding(${vocab}, ${embed})(${mainIn})`;
    }

    case "BatchNorm":
      return `keras.layers.BatchNormalization()(${mainIn})`;

    case "LayerNorm":
      return `keras.layers.LayerNormalization()(${mainIn})`;

    case "MaxPool": {
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 2;
      const stride = getNum(block, "stride") ?? kernel;
      return `keras.layers.MaxPooling2D(pool_size=${kernel}, strides=${stride})(${mainIn})`;
    }

    case "AvgPool": {
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 2;
      const stride = getNum(block, "stride") ?? kernel;
      return `keras.layers.AveragePooling2D(pool_size=${kernel}, strides=${stride})(${mainIn})`;
    }

    case "GlobalAvgPool":
      return `keras.layers.GlobalAveragePooling2D()(${mainIn})`;

    case "Dropout": {
      const p = getNum(block, "p") ?? getNum(block, "rate") ?? 0.5;
      return `keras.layers.Dropout(${p})(${mainIn})`;
    }

    case "Flatten":
      return `keras.layers.Flatten()(${mainIn})`;

    case "LSTM": {
      const hidden = getNum(block, "hidden_size") ?? getNum(block, "hidden") ?? 0;
      return `keras.layers.LSTM(${hidden}, return_sequences=True)(${mainIn})`;
    }

    case "GRU": {
      const hidden = getNum(block, "hidden_size") ?? getNum(block, "hidden") ?? 0;
      return `keras.layers.GRU(${hidden}, return_sequences=True)(${mainIn})`;
    }

    case "Add":
      return `keras.layers.Add()([${inputVars.join(", ")}])`;

    case "Concat": {
      const axis = getNum(block, "axis") ?? -1;
      return `keras.layers.Concatenate(axis=${axis})([${inputVars.join(", ")}])`;
    }

    case "Reshape": {
      const shapeParam = block.params["shape"];
      if (shapeParam && shapeParam.kind === "shape") {
        const dims = shapeParam.dims.join(", ");
        return `keras.layers.Reshape((${dims},))(${mainIn})`;
      }
      return `keras.layers.Reshape((-1,))(${mainIn})`;
    }

    case "ReLU":
      return `keras.layers.ReLU()(${mainIn})`;
    case "LeakyReLU": {
      const slope = getNum(block, "negative_slope") ?? 0.01;
      return `keras.layers.LeakyReLU(alpha=${slope})(${mainIn})`;
    }
    case "ELU": {
      const alpha = getNum(block, "alpha") ?? 1.0;
      return `keras.layers.ELU(alpha=${alpha})(${mainIn})`;
    }
    case "Sigmoid":
      return `keras.layers.Activation('sigmoid')(${mainIn})`;
    case "Tanh":
      return `keras.layers.Activation('tanh')(${mainIn})`;
    case "Softmax": {
      const axis = getNum(block, "dim") ?? -1;
      return `keras.layers.Softmax(axis=${axis})(${mainIn})`;
    }
    case "GELU":
      return `keras.layers.Activation('gelu')(${mainIn})`;
    case "SiLU":
      return `keras.layers.Activation('swish')(${mainIn})`;
    case "PReLU":
      return `keras.layers.PReLU()(${mainIn})`;

    default:
      return `keras.layers.Lambda(lambda x: x)(${mainIn})  # unknown block type: ${t}`;
  }
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

export function generateKeras(graph: Graph, _registry: Map<string, BlockDef>): GeneratedFile[] {
  const adj = buildAdjacency(graph);
  const sorted = topoSort(graph);

  // Determine named outputs (from edge tensorName)
  const namedOutputs = new Map<string, string>(); // blockId → variableName
  for (const e of graph.edges) {
    if (e.tensorName) {
      namedOutputs.set(e.from, e.tensorName);
    }
  }

  const lines: string[] = [
    `import tensorflow as tf`,
    `from tensorflow import keras`,
    ``,
    ``,
    `def build_model():`,
  ];

  // Track what variable name each block's output is stored in
  const blockOutputVar = new Map<string, string>(); // blockId → python var name

  let varCounter = 0;
  const freshVar = () => {
    varCounter++;
    return varCounter === 1 ? "x" : `x${varCounter}`;
  };

  let inputVar = "inputs";
  let outputVar = "x";

  for (const b of sorted) {
    const info = adj.get(b.id)!;
    const inputVars: string[] = info.inputs.map((inId) => blockOutputVar.get(inId) ?? "x");

    if (b.type === "Input") {
      // Generate keras.Input
      const shape = b.outputShapes[0] ?? [];
      // Drop batch dimension
      const spatialShape = shape.slice(1);
      const shapeStr = spatialShape.length ? spatialShape.join(", ") : "None";
      const named = namedOutputs.get(b.id);
      const outVar = named ?? inputVar;
      blockOutputVar.set(b.id, outVar);
      const shapeAnn = shapeComment(b.outputShapes);
      lines.push(`    ${outVar} = keras.Input(shape=(${shapeStr},))${shapeAnn}`);
      inputVar = outVar;
      continue;
    }

    if (b.type === "Output") {
      const retVar = inputVars[0] ?? "x";
      outputVar = retVar;
      const shapeAnn = shapeComment(b.inputShapes);
      lines.push(`    # output${shapeAnn}`);
      continue;
    }

    const expr = blockToLayerExpr(b, inputVars);
    if (!expr) continue;

    // Determine output variable name
    let outVar: string;
    if (namedOutputs.has(b.id)) {
      outVar = namedOutputs.get(b.id)!;
    } else if (info.outputs.length <= 1) {
      outVar = "x";
    } else {
      outVar = freshVar();
    }
    blockOutputVar.set(b.id, outVar);

    const shapeAnn = shapeComment(b.outputShapes);
    lines.push(`    ${outVar} = ${expr}${shapeAnn}`);
  }

  lines.push(`    return keras.Model(inputs=${inputVar}, outputs=${outputVar})`);

  const modelName = graph.groups[0]?.path[0] ?? "Model";
  const content = lines.join("\n") + "\n";
  return [{ path: `${modelName}.py`, content }];
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

registerTarget({
  name: "keras",
  fileExtension: ".py",
  generate: generateKeras,
});
