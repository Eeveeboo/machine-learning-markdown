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
// Layer attribute generation (for __init__)
// ---------------------------------------------------------------------------

interface LayerAttr {
  attrName: string;
  initExpr: string;
}

function blockToAttr(block: Block): LayerAttr | null {
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
    case "Pad":
      return null;

    case "Linear": {
      const inF = inputShape[inputShape.length - 1] ?? 0;
      const outF = getNum(block, "out_features") ?? getNum(block, "units") ?? outputShape[outputShape.length - 1] ?? 0;
      return { attrName: block.id, initExpr: `nn.Linear(${inF}, ${outF})` };
    }

    case "Conv2d": {
      const inCh = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
      const filters = getNum(block, "filters") ?? getNum(block, "out_channels") ?? 0;
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 3;
      const stride = getNum(block, "stride") ?? 1;
      const padding = getNum(block, "padding") ?? 0;
      return { attrName: block.id, initExpr: `nn.Conv2d(${inCh}, ${filters}, ${kernel}, stride=${stride}, padding=${padding})` };
    }

    case "Conv1d": {
      const inCh = inputShape[inputShape.length - 2] ?? inputShape[1] ?? 0;
      const filters = getNum(block, "filters") ?? getNum(block, "out_channels") ?? 0;
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 3;
      const stride = getNum(block, "stride") ?? 1;
      const padding = getNum(block, "padding") ?? 0;
      return { attrName: block.id, initExpr: `nn.Conv1d(${inCh}, ${filters}, ${kernel}, stride=${stride}, padding=${padding})` };
    }

    case "Conv3d": {
      const inCh = inputShape[inputShape.length - 4] ?? inputShape[1] ?? 0;
      const filters = getNum(block, "filters") ?? getNum(block, "out_channels") ?? 0;
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 3;
      const stride = getNum(block, "stride") ?? 1;
      const padding = getNum(block, "padding") ?? 0;
      return { attrName: block.id, initExpr: `nn.Conv3d(${inCh}, ${filters}, ${kernel}, stride=${stride}, padding=${padding})` };
    }

    case "TransposedConv2d": {
      const inCh = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
      const filters = getNum(block, "filters") ?? getNum(block, "out_channels") ?? 0;
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 3;
      const stride = getNum(block, "stride") ?? 1;
      const padding = getNum(block, "padding") ?? 0;
      return { attrName: block.id, initExpr: `nn.ConvTranspose2d(${inCh}, ${filters}, ${kernel}, stride=${stride}, padding=${padding})` };
    }

    case "Embedding": {
      const vocab = getNum(block, "vocab_size") ?? 0;
      const embed = getNum(block, "embed_dim") ?? getNum(block, "embedding_dim") ?? 0;
      return { attrName: block.id, initExpr: `nn.Embedding(${vocab}, ${embed})` };
    }

    case "BatchNorm": {
      const dims = inputShape.length;
      if (dims <= 2) {
        const num = inputShape[inputShape.length - 1] ?? 0;
        return { attrName: block.id, initExpr: `nn.BatchNorm1d(${num})` };
      } else {
        const num = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
        return { attrName: block.id, initExpr: `nn.BatchNorm2d(${num})` };
      }
    }

    case "LayerNorm": {
      const normalized = inputShape.length ? `[${inputShape.slice(1).join(", ")}]` : "[]";
      return { attrName: block.id, initExpr: `nn.LayerNorm(${normalized})` };
    }

    case "GroupNorm": {
      const groups = getNum(block, "num_groups") ?? 32;
      const num = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
      return { attrName: block.id, initExpr: `nn.GroupNorm(${groups}, ${num})` };
    }

    case "InstanceNorm": {
      const dims = inputShape.length;
      if (dims <= 2) {
        const num = inputShape[inputShape.length - 1] ?? 0;
        return { attrName: block.id, initExpr: `nn.InstanceNorm1d(${num})` };
      } else {
        const num = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
        return { attrName: block.id, initExpr: `nn.InstanceNorm2d(${num})` };
      }
    }

    case "MaxPool": {
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 2;
      const stride = getNum(block, "stride") ?? kernel;
      return { attrName: block.id, initExpr: `nn.MaxPool2d(${kernel}, stride=${stride})` };
    }

    case "AvgPool": {
      const kernel = getNum(block, "kernel") ?? getNum(block, "kernel_size") ?? 2;
      const stride = getNum(block, "stride") ?? kernel;
      return { attrName: block.id, initExpr: `nn.AvgPool2d(${kernel}, stride=${stride})` };
    }

    case "GlobalAvgPool":
      return { attrName: block.id, initExpr: `nn.AdaptiveAvgPool2d(1)` };

    case "AdaptiveAvgPool": {
      const outH = getNum(block, "output_size_h") ?? getNum(block, "output_size") ?? 1;
      const outW = getNum(block, "output_size_w") ?? outH;
      return { attrName: block.id, initExpr: `nn.AdaptiveAvgPool2d((${outH}, ${outW}))` };
    }

    case "Dropout": {
      const p = getNum(block, "p") ?? getNum(block, "rate") ?? 0.5;
      return { attrName: block.id, initExpr: `nn.Dropout(p=${p})` };
    }

    case "Flatten":
      return { attrName: block.id, initExpr: `nn.Flatten()` };

    case "LSTM": {
      const inputSize = inputShape[inputShape.length - 1] ?? 0;
      const hidden = getNum(block, "hidden_size") ?? getNum(block, "hidden") ?? 0;
      const layers = getNum(block, "num_layers") ?? 1;
      const batchFirst = true;
      return { attrName: block.id, initExpr: `nn.LSTM(${inputSize}, ${hidden}, num_layers=${layers}, batch_first=${batchFirst})` };
    }

    case "GRU": {
      const inputSize = inputShape[inputShape.length - 1] ?? 0;
      const hidden = getNum(block, "hidden_size") ?? getNum(block, "hidden") ?? 0;
      const layers = getNum(block, "num_layers") ?? 1;
      return { attrName: block.id, initExpr: `nn.GRU(${inputSize}, ${hidden}, num_layers=${layers}, batch_first=True)` };
    }

    // Activations stored as module attrs
    case "ReLU":
      return { attrName: block.id, initExpr: `nn.ReLU()` };
    case "LeakyReLU": {
      const slope = getNum(block, "negative_slope") ?? 0.01;
      return { attrName: block.id, initExpr: `nn.LeakyReLU(${slope})` };
    }
    case "PReLU":
      return { attrName: block.id, initExpr: `nn.PReLU()` };
    case "ELU": {
      const alpha = getNum(block, "alpha") ?? 1.0;
      return { attrName: block.id, initExpr: `nn.ELU(alpha=${alpha})` };
    }
    case "Sigmoid":
      return { attrName: block.id, initExpr: `nn.Sigmoid()` };
    case "Tanh":
      return { attrName: block.id, initExpr: `nn.Tanh()` };
    case "Softmax": {
      const dim = getNum(block, "dim") ?? -1;
      return { attrName: block.id, initExpr: `nn.Softmax(dim=${dim})` };
    }
    case "GELU":
      return { attrName: block.id, initExpr: `nn.GELU()` };
    case "SiLU":
      return { attrName: block.id, initExpr: `nn.SiLU()` };

    default:
      // Unknown block: emit a placeholder comment
      return { attrName: block.id, initExpr: `nn.Identity()  # unknown block type: ${t}` };
  }
}

// ---------------------------------------------------------------------------
// Forward pass expression generation
// ---------------------------------------------------------------------------

/** Returns the Python expression for one block applied to its input variable(s). */
function blockToForwardExpr(block: Block, inputVars: string[]): string[] {
  const t = block.type;
  const mainIn = inputVars[0] ?? "x";

  switch (t) {
    case "Input":
      return [`${mainIn}`];
    case "Output":
      return [`return ${mainIn}`];

    case "Add":
      return [`${inputVars.join(" + ")}`];
    case "Mul":
      return [`${inputVars.join(" * ")}`];
    case "Concat": {
      const dim = 1;
      return [`torch.cat([${inputVars.join(", ")}], dim=${dim})`];
    }

    case "Reshape": {
      const shapeParam = block.params["shape"];
      if (shapeParam && shapeParam.kind === "shape") {
        const dims = shapeParam.dims.join(", ");
        return [`${mainIn}.reshape(${mainIn}.size(0), ${dims})`];
      }
      return [`${mainIn}.reshape(${mainIn}.size(0), -1)`];
    }

    case "Pad": {
      const padding = block.params["padding"];
      if (padding && padding.kind === "list") {
        const vals = padding.items
          .filter((i) => i.kind === "number")
          .map((i) => (i as { kind: "number"; value: number }).value);
        return [`torch.nn.functional.pad(${mainIn}, (${vals.join(", ")}))`];
      }
      return [`torch.nn.functional.pad(${mainIn}, (0, 0))`];
    }

    case "LSTM":
    case "GRU":
      // These return (output, hidden); we use output only
      return [`self.${block.id}(${mainIn})[0]`];

    default:
      return [`self.${block.id}(${mainIn})`];
  }
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

export function generatePytorch(graph: Graph, _registry: Map<string, BlockDef>): GeneratedFile[] {
  const adj = buildAdjacency(graph);
  const sorted = topoSort(graph);

  // Build edge tensorName lookup: edge.from+edge.to → tensorName
  const edgeTensorName = new Map<string, string>();
  for (const e of graph.edges) {
    if (e.tensorName) {
      edgeTensorName.set(`${e.from}->${e.to}`, e.tensorName);
    }
  }

  // Determine which block outputs have named tensors (saved to variable)
  // A block's output is "named" if any outgoing edge has a tensorName
  const namedOutputs = new Map<string, string>(); // blockId → variableName
  for (const e of graph.edges) {
    if (e.tensorName) {
      namedOutputs.set(e.from, e.tensorName);
    }
  }

  // __init__ lines
  const initLines: string[] = [];
  const attrs = new Set<string>();
  for (const b of sorted) {
    const attr = blockToAttr(b);
    if (attr && !attrs.has(attr.attrName)) {
      attrs.add(attr.attrName);
      initLines.push(`        self.${attr.attrName} = ${attr.initExpr}`);
    }
  }

  // forward lines
  const forwardLines: string[] = [];

  // Track what variable name each block's output is stored in
  const blockOutputVar = new Map<string, string>(); // blockId → python var name

  // Assign initial "x" to first block's output
  let varCounter = 0;
  const freshVar = () => {
    varCounter++;
    return varCounter === 1 ? "x" : `x${varCounter}`;
  };

  for (const b of sorted) {
    const info = adj.get(b.id)!;

    // Determine input variable names for this block
    const inputVars: string[] = info.inputs.map((inId) => blockOutputVar.get(inId) ?? "x");

    if (b.type === "Input") {
      const named = namedOutputs.get(b.id);
      const outVar = named ?? "x";
      blockOutputVar.set(b.id, outVar);
      const inShapeComment = shapeComment(b.outputShapes);
      if (named) {
        forwardLines.push(`        ${named} = x${inShapeComment !== "" ? "  " + inShapeComment.trim() : ""}`);
      } else {
        forwardLines.push(`        # x: input${inShapeComment}`);
      }
      continue;
    }

    if (b.type === "Output") {
      const retVar = inputVars[0] ?? "x";
      const retShapeComment = shapeComment(b.inputShapes);
      forwardLines.push(`        return ${retVar}${retShapeComment !== "" ? "  " + retShapeComment.trim() : ""}`);
      continue;
    }

    const exprs = blockToForwardExpr(b, inputVars);
    const outExpr = exprs[0];

    // Determine output variable name
    let outVar: string;
    if (namedOutputs.has(b.id)) {
      outVar = namedOutputs.get(b.id)!;
    } else if (info.outputs.length === 1) {
      // If only one consumer, reuse "x" pattern unless we need a named tensor
      outVar = "x";
    } else {
      outVar = freshVar();
    }
    blockOutputVar.set(b.id, outVar);

    const shapeAnn = shapeComment(b.outputShapes);
    forwardLines.push(`        ${outVar} = ${outExpr}${shapeAnn}`);
  }

  // Class name from graph groups or default
  const className = graph.groups[0]?.path[0] ?? "Model";

  const lines: string[] = [
    `import torch`,
    `import torch.nn as nn`,
    `import torch.nn.functional as F`,
    ``,
    ``,
    `class ${className}(nn.Module):`,
    `    def __init__(self):`,
    `        super().__init__()`,
    ...initLines,
    ``,
    `    def forward(self, x):`,
    ...forwardLines,
  ];

  const content = lines.join("\n") + "\n";
  return [{ path: `${className}.py`, content }];
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

registerTarget({
  name: "pytorch",
  fileExtension: ".py",
  generate: generatePytorch,
});
