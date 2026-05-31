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

// ---------------------------------------------------------------------------
// Struct field generation (learnable layers only)
// ---------------------------------------------------------------------------

interface StructField {
  fieldName: string;
  fieldType: string;
  initExpr: string;
}

function blockToField(block: Block): StructField | null {
  const t = block.type;
  const inputShape = block.inputShapes[0] ?? [];
  const outputShape = block.outputShapes[0] ?? [];

  switch (t) {
    case "Input":
    case "Output":
    case "Add":
    case "Mul":
    case "Concat":
    case "Reshape":
    case "Flatten":
    case "ReLU":
    case "Sigmoid":
    case "Tanh":
    case "GELU":
    case "SiLU":
    case "MaxPool":
    case "GlobalAvgPool":
      return null;

    case "Linear": {
      const inF = inputShape[inputShape.length - 1] ?? 0;
      const outF = getNum(block, "out_features") ?? getNum(block, "units") ?? outputShape[outputShape.length - 1] ?? 0;
      return {
        fieldName: block.id,
        fieldType: "candle_nn::Linear",
        initExpr: `candle_nn::linear(${inF}, ${outF}, vb.pp("${block.id}"))?`,
      };
    }

    case "Conv2d": {
      const inCh = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
      const filters = getNum(block, "filters") ?? getNum(block, "out_channels") ?? 0;
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 3;
      const stride = getNum(block, "stride") ?? 1;
      const padding = getNum(block, "padding") ?? 0;
      return {
        fieldName: block.id,
        fieldType: "candle_nn::Conv2d",
        initExpr: `candle_nn::conv2d(${inCh}, ${filters}, ${kernel}, candle_nn::Conv2dConfig { stride: ${stride}, padding: ${padding}, ..Default::default() }, vb.pp("${block.id}"))?`,
      };
    }

    case "BatchNorm": {
      const features = inputShape.length >= 4
        ? (inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0)
        : (inputShape[inputShape.length - 1] ?? 0);
      return {
        fieldName: block.id,
        fieldType: "candle_nn::BatchNorm",
        initExpr: `candle_nn::batch_norm(${features}, 1e-5, vb.pp("${block.id}"))?`,
      };
    }

    case "LayerNorm": {
      const features = inputShape[inputShape.length - 1] ?? 0;
      return {
        fieldName: block.id,
        fieldType: "candle_nn::LayerNorm",
        initExpr: `candle_nn::layer_norm(${features}, 1e-5, vb.pp("${block.id}"))?`,
      };
    }

    case "Dropout": {
      const p = getNum(block, "p") ?? getNum(block, "rate") ?? 0.5;
      return {
        fieldName: block.id,
        fieldType: "candle_nn::Dropout",
        initExpr: `candle_nn::Dropout::new(${p})`,
      };
    }

    case "Embedding": {
      const vocab = getNum(block, "vocab_size") ?? 0;
      const dim = getNum(block, "embed_dim") ?? getNum(block, "embedding_dim") ?? 0;
      return {
        fieldName: block.id,
        fieldType: "candle_nn::Embedding",
        initExpr: `candle_nn::embedding(${vocab}, ${dim}, vb.pp("${block.id}"))?`,
      };
    }

    case "LSTM": {
      const inputSize = inputShape[inputShape.length - 1] ?? 0;
      const hidden = getNum(block, "hidden_size") ?? getNum(block, "hidden") ?? 0;
      return {
        fieldName: block.id,
        fieldType: "candle_nn::LSTM",
        initExpr: `candle_nn::lstm(${inputSize}, ${hidden}, candle_nn::LSTMConfig::default(), vb.pp("${block.id}"))?`,
      };
    }

    case "GRU": {
      const inputSize = inputShape[inputShape.length - 1] ?? 0;
      const hidden = getNum(block, "hidden_size") ?? getNum(block, "hidden") ?? 0;
      return {
        fieldName: block.id,
        fieldType: "candle_nn::GRU",
        initExpr: `candle_nn::gru(${inputSize}, ${hidden}, candle_nn::GRUConfig::default(), vb.pp("${block.id}"))?`,
      };
    }

    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// Forward expression generation
// ---------------------------------------------------------------------------

function blockToForwardExpr(block: Block, inputVars: string[]): string {
  const t = block.type;
  const mainIn = inputVars[0] ?? "x";

  switch (t) {
    case "ReLU":
      return `${mainIn}.relu()?`;
    case "Sigmoid":
      return `${mainIn}.sigmoid()?`;
    case "Tanh":
      return `${mainIn}.tanh()?`;
    case "GELU":
      return `${mainIn}.gelu()?`;
    case "SiLU":
      return `${mainIn}.silu()?`;

    case "Flatten":
      return `${mainIn}.flatten_from(1)?`;

    case "MaxPool": {
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 2;
      const stride = getNum(block, "stride") ?? kernel;
      return `candle_nn::ops::max_pool2d(&${mainIn}, ${kernel}, ${stride})?`;
    }

    case "GlobalAvgPool":
      return `${mainIn}.mean_keepdim(candle_core::D::Minus1)?.mean_keepdim(candle_core::D::Minus2)?`;

    case "Add":
      return `(&${inputVars[0] ?? "x"} + &${inputVars[1] ?? "x"})?`;

    case "Concat": {
      const tensors = inputVars.map((v) => `&${v}`).join(", ");
      return `Tensor::cat(&[${tensors}], 1)?`;
    }

    case "Dropout":
      return `self.${block.id}.forward(&${mainIn}, true)?`;

    case "BatchNorm":
    case "LayerNorm":
      return `self.${block.id}.forward(&${mainIn})?`;

    case "LSTM": {
      // candle LSTM: returns sequence of states; use last hidden
      return `{ let states = candle_nn::RNN::seq(&self.${block.id}, &${mainIn})?; candle_nn::RNN::states_to_tensor(&self.${block.id}, &states)? }`;
    }

    case "GRU": {
      return `{ let states = candle_nn::RNN::seq(&self.${block.id}, &${mainIn})?; candle_nn::RNN::states_to_tensor(&self.${block.id}, &states)? }`;
    }

    default:
      // learnable layers use Module::forward
      return `self.${block.id}.forward(&${mainIn})?`;
  }
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

export function generateCandle(graph: Graph, _registry: Map<string, BlockDef>): GeneratedFile[] {
  const adj = buildAdjacency(graph);
  const sorted = topoSort(graph);

  const namedOutputs = new Map<string, string>();
  for (const e of graph.edges) {
    if (e.tensorName) {
      namedOutputs.set(e.from, e.tensorName);
    }
  }

  // Collect struct fields (learnable layers)
  const fields: StructField[] = [];
  const seen = new Set<string>();
  for (const b of sorted) {
    const field = blockToField(b);
    if (field && !seen.has(field.fieldName)) {
      seen.add(field.fieldName);
      fields.push(field);
    }
  }

  // Track output variable per block
  const blockOutputVar = new Map<string, string>();
  let varCounter = 0;
  const freshVar = () => {
    varCounter++;
    return varCounter === 1 ? "x" : `x${varCounter}`;
  };

  const forwardLines: string[] = [];

  for (const b of sorted) {
    const info = adj.get(b.id)!;
    const inputVars: string[] = info.inputs.map((inId) => blockOutputVar.get(inId) ?? "x");

    if (b.type === "Input") {
      const named = namedOutputs.get(b.id);
      blockOutputVar.set(b.id, named ?? "x");
      continue;
    }

    if (b.type === "Output") {
      const retVar = inputVars[0] ?? "x";
      forwardLines.push(`        Ok(${retVar})`);
      continue;
    }

    const expr = blockToForwardExpr(b, inputVars);

    let outVar: string;
    if (namedOutputs.has(b.id)) {
      outVar = namedOutputs.get(b.id)!;
    } else if (info.outputs.length === 1) {
      outVar = "x";
    } else {
      outVar = freshVar();
    }
    blockOutputVar.set(b.id, outVar);

    forwardLines.push(`        let ${outVar} = ${expr};`);
  }

  const className = graph.groups[0]?.path[0] ?? "Model";

  // Struct fields
  const structFieldLines = fields.map((f) => `    ${f.fieldName}: ${f.fieldType},`);
  // new fn
  const newInitLines = fields.map((f) => `        let ${f.fieldName} = ${f.initExpr};`);
  const newOkFields = fields.map((f) => `            ${f.fieldName},`);

  const lines: string[] = [
    `use candle_core::{Result, Tensor};`,
    `use candle_nn::{Module, VarBuilder};`,
    ``,
    `pub struct ${className} {`,
    ...structFieldLines,
    `}`,
    ``,
    `impl ${className} {`,
    `    pub fn new(vb: VarBuilder) -> Result<Self> {`,
    ...newInitLines,
    `        Ok(Self {`,
    ...newOkFields,
    `        })`,
    `    }`,
    ``,
    `    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {`,
    ...forwardLines,
    `    }`,
    `}`,
  ];

  const content = lines.join("\n") + "\n";
  return [{ path: `${className}.rs`, content }];
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

registerTarget({
  name: "candle",
  fileExtension: ".rs",
  generate: generateCandle,
});
