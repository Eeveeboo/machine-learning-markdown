import type { Graph, Block } from "../../ast/graph.js";
import type { BlockDef } from "../../blocks/types.js";
import { registerTarget } from "../target.js";
import type { GeneratedFile } from "../target.js";
import { topoSort, buildAdjacency } from "../../ast/graph-utils.js";
import { getBlockCodegenWithFallback } from "../block-codegen.js";

// Ensure all built-in block codegen functions are registered before any codegen runs
import "../../plugins/builtins/index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function shapeComment(shapes: number[][]): string {
  if (!shapes.length) return "";
  return "  # " + shapes.map((s) => `[${s.join(", ")}]`).join(", ");
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
    if (b.type === "Input" || b.type === "Output") continue;
    const fn = getBlockCodegenWithFallback(b.type, "pytorch");
    const result = fn(b, []);
    if (result.attr && !attrs.has(result.attr.name)) {
      attrs.add(result.attr.name);
      initLines.push(`        self.${result.attr.name} = ${result.attr.init}`);
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

    const fn = getBlockCodegenWithFallback(b.type, "pytorch");
    const result = fn(b, inputVars);
    const outExpr = result.forward;

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
