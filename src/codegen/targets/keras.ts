import type { Graph } from "../../ast/graph.js";
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

    const fn = getBlockCodegenWithFallback(b.type, "keras");
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

    const outCount = b.outputShapes.length;
    let outputVars: string[];
    if (outCount <= 1) {
      outputVars = [outVar];
    } else {
      const base = freshVar();
      outputVars = Array.from({ length: outCount }, (_, i) => `${base}_${i}`);
    }
    const result = fn(b, inputVars, outputVars);
    const shapeAnn = shapeComment(b.outputShapes);
    lines.push(`    ${result.forward}${shapeAnn}`);
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
