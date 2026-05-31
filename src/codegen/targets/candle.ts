import type { Graph } from "../../ast/graph.js";
import type { BlockDef } from "../../blocks/types.js";
import { registerTarget } from "../target.js";
import type { GeneratedFile } from "../target.js";
import { topoSort, buildAdjacency } from "../../ast/graph-utils.js";
import { getBlockCodegenWithFallback } from "../block-codegen.js";

// Ensure all built-in block codegen functions are registered before any codegen runs
import "../../plugins/builtins/index.js";

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

  // Collect struct fields (learnable layers) using plugin codegen
  interface StructField {
    fieldName: string;
    fieldType: string;
    initExpr: string;
  }

  const fields: StructField[] = [];
  const seen = new Set<string>();
  for (const b of sorted) {
    if (b.type === "Input" || b.type === "Output") continue;
    const fn = getBlockCodegenWithFallback(b.type, "candle");
    const result = fn(b, []);
    if (result.attr && !seen.has(result.attr.name)) {
      seen.add(result.attr.name);
      fields.push({
        fieldName: result.attr.name,
        fieldType: result.attr.typeAnnotation ?? "/* unknown */",
        initExpr: result.attr.init,
      });
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

    const fn = getBlockCodegenWithFallback(b.type, "candle");
    const result = fn(b, inputVars);
    const expr = result.forward;

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
