import type { Graph } from "../../ast/graph.js";
import type { BlockDef } from "../../blocks/types.js";
import { registerTarget } from "../target.js";
import type { GeneratedFile } from "../target.js";
import { topoSort, buildAdjacency } from "../../ast/graph-utils.js";
import { getBlockCodegenWithFallback } from "../block-codegen.js";

// Ensure all built-in block codegen functions are registered before any codegen runs
import "../../plugins/builtins/index.js";

/** Convert PascalCase or TitleCase field name to snake_case Rust param name */
function toSnakeCase(name: string): string {
  return name.charAt(0).toLowerCase() + name.slice(1);
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

  // Collect Input blocks → forward parameter names
  const inputParams = new Map<string, string>(); // blockId → paramName
  const forwardParams: string[] = [];
  let unnamedCount = 0;

  for (const b of sorted) {
    if (b.type !== "Input") continue;
    const named = namedOutputs.get(b.id);
    if (named) {
      inputParams.set(b.id, named);
      forwardParams.push(`${named}: &Tensor`);
    } else {
      unnamedCount++;
      const name = unnamedCount === 1 ? "x" : `x${unnamedCount}`;
      inputParams.set(b.id, name);
      forwardParams.push(`${name}: &Tensor`);
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
      blockOutputVar.set(b.id, inputParams.get(b.id) ?? "x");
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

  // Constructors
  // -- with_scopes: custom &str param per field for weight-loading scope names
  const scopeParams = fields.map((f) => `        ${toSnakeCase(f.fieldName)}: &str,`);
  const withScopesInitLines = fields.map((f) => {
    const scopeParam = toSnakeCase(f.fieldName);
    // Replace vb.pp("FieldName") from plugin init with weights.pp(param) reference
    const expr = f.initExpr.replace(`vb.pp("${f.fieldName}")`, `weights.pp(${scopeParam})`);
    return `        let ${f.fieldName} = ${expr};`;
  });
  const newOkFields = fields.map((f) => `            ${f.fieldName},`);
  // -- new: convenience that auto-generates scope names from field names
  const defaultScopeArgs = fields.map((f) => `        "${f.fieldName}",`);

  // Forward signature
  const forwardSignature =
    forwardParams.length > 0
      ? forwardParams.join(",\n        ")
      : "// no inputs defined";
  const forwardParamBlock =
    forwardParams.length > 0
      ? `        ${forwardSignature}`
      : "        // no inputs defined";

  const lines: string[] = [
    `use candle_core::{Result, Tensor};`,
    `use candle_nn::{Module, VarBuilder};`,
    ``,
    `pub struct ${className} {`,
    ...structFieldLines,
    `}`,
    ``,
    `impl ${className} {`,
    `    /// Create model with auto-generated weight scope names.`,
    `    /// Use [${className}::with_scopes] for custom scope names.`,
    `    pub fn new(weights: VarBuilder) -> Result<Self> {`,
    `        Self::with_scopes(weights,`,
    ...defaultScopeArgs,
    `        )`,
    `    }`,
    ``,
    `    /// Create model with custom weight-loading scope names.`,
    `    pub fn with_scopes(`,
    `        weights: VarBuilder,`,
    ...scopeParams,
    `    ) -> Result<Self> {`,
    ...withScopesInitLines,
    `        Ok(Self {`,
    ...newOkFields,
    `        })`,
    `    }`,
    ``,
    `    pub fn forward(&self,`,
    forwardParamBlock,
    `    ) -> Result<Tensor> {`,
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
