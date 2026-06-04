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

function indentLines(str: string, indent: string): string {
  return str.split("\n").map((line) => line ? `${indent}${line}` : line).join("\n");
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
  for (const b of sorted) {
    if (b.type === "Input" || b.type === "Output") continue;
    const fn = getBlockCodegenWithFallback(b.type, "pytorch");
    const result = fn(b, [], []);
    if (typeof result.init === "string") {
      initLines.push(indentLines(result.init, "        "));
    }
  }

  // Collect Input blocks → forward parameter names (like candle codegen)
  const inputParams = new Map<string, string>(); // blockId → paramName
  const forwardParams: string[] = [];
  let unnamedCount = 0;

  for (const b of sorted) {
    if (b.type !== "Input") continue;
    const named = namedOutputs.get(b.id);
    if (named) {
      inputParams.set(b.id, named);
      forwardParams.push(named);
    } else {
      unnamedCount++;
      const name = unnamedCount === 1 ? "x" : `x${unnamedCount}`;
      inputParams.set(b.id, name);
      forwardParams.push(name);
    }
  }

  // forward lines
  const forwardLines: string[] = [];

  // Track what variable name each block's output is stored in
  const blockOutputVar = new Map<string, string>(); // blockId → python var name

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
      const paramName = inputParams.get(b.id) ?? "x";
      blockOutputVar.set(b.id, paramName);
      const inShapeComment = shapeComment(b.outputShapes);
      if (inShapeComment) {
        forwardLines.push(`        # ${paramName}: input${inShapeComment}`);
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
    const outCount = b.outputShapes.length;
    let outputVars: string[];
    if (outCount <= 1) {
      if (namedOutputs.has(b.id)) {
        outputVars = [namedOutputs.get(b.id)!];
      } else if (info.outputs.length === 1) {
        outputVars = ["x"];
      } else {
        outputVars = [freshVar()];
      }
    } else {
      // Multi-output: generate N names with suffix
      const base = freshVar();
      outputVars = Array.from({ length: outCount }, (_, i) => `${base}_${i}`);
    }
    blockOutputVar.set(b.id, outputVars[0]);

    const result = fn(b, inputVars, outputVars);
    const shapeAnn = shapeComment(b.outputShapes);
    forwardLines.push(indentLines(result.forward, "        ") + shapeAnn);
  }

  // Class name from graph groups or default
  const className = graph.groups[0]?.path[0] ?? "Model";

  // Forward signature with proper multi-input params
  const forwardSig =
    forwardParams.length > 0
      ? forwardParams.join(", ")
      : "x";

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
    `    def forward(self, ${forwardSig}):`,
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
