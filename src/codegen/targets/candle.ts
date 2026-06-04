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

function indentLines(str: string, indent: string): string {
  return str.split("\n").map((line) => line ? `${indent}${line}` : line).join("\n");
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
  const structFieldLines: string[] = [];
  const withScopesInitLines: string[] = [];
  const newOkFields: string[] = [];
  const defaultScopeArgs: string[] = [];
  const scopeParams: string[] = [];

  for (const b of sorted) {
    if (b.type === "Input" || b.type === "Output") continue;
    const fn = getBlockCodegenWithFallback(b.type, "candle");
    const result = fn(b, [], []);  // init pass: no var names needed
    if (result.init && typeof result.init === "object" && "field" in result.init) {
      const candleInit = result.init as { field: string; body: string };
      structFieldLines.push(`    ${candleInit.field}`);

      // Transform vb.pp("blockId") → weights.pp(snakeName) for with_scopes
      const snakeName = toSnakeCase(b.id);
      const body = candleInit.body.replace(
        new RegExp(`vb\\.pp\\("${b.id}"\\)`, "g"),
        `weights.pp(${snakeName})`
      );
      withScopesInitLines.push(indentLines(body, "        "));

      newOkFields.push(`            ${b.id},`);
      defaultScopeArgs.push(`        "${b.id}",`);
      scopeParams.push(`        ${toSnakeCase(candleInit.field.split(":")[0].trim())}: &str,`);
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
      const base = freshVar();
      outputVars = Array.from({ length: outCount }, (_, i) => `${base}_${i}`);
    }
    blockOutputVar.set(b.id, outputVars[0]);

    const result = fn(b, inputVars, outputVars);
    forwardLines.push(indentLines(result.forward, "        "));
  }

  const className = graph.groups[0]?.path[0] ?? "Model";

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
