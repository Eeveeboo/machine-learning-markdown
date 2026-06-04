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
// Test module generation
// ---------------------------------------------------------------------------

/**
 * Generate a `#[cfg(test)] mod tests { ... }` block for Rust Candle models.
 * Contains test_forward (validates forward pass output shape) and test_save_load
 * (validates weight save/load round-trip produces identical outputs).
 *
 * Returns an empty string if outputShape is null or forwardParams is empty.
 */
function generateTestModule(
  className: string,
  forwardParams: string[],
  inputShapes: number[][],
  outputShape: number[] | null,
): string {
  // Edge cases: skip if no output shape or no forward params
  if (outputShape === null || forwardParams.length === 0) {
    return "";
  }

  // Extract param variable names: "x: &Tensor" → "x"
  const paramNames = forwardParams.map(p => p.split(":")[0].trim());

  // Determine which shape to use for each param.
  // Defensive: if lengths mismatch, use inputShapes[0] for all params.
  const shapesForParams: number[][] = [];
  if (forwardParams.length !== inputShapes.length) {
    for (let i = 0; i < forwardParams.length; i++) {
      shapesForParams.push(inputShapes[0] ?? []);
    }
  } else {
    shapesForParams.push(...inputShapes);
  }

  // Helper: format number[] as Rust slice literal (for Tensor::randn and assert_eq)
  //   [1, 1, 28, 28] → &[1, 1, 28, 28]
  function formatShapeSlice(shape: number[]): string {
    if (shape.length === 0) return "&[]";
    return `&[${shape.join(", ")}]`;
  }

  // Build `let param = Tensor::randn(0f32, 1.0, &[shape], &dev)?;` lines
  const inputTensorLines = paramNames.map((name, i) => {
    const shapeSlice = formatShapeSlice(shapesForParams[i]);
    return `        let ${name} = candle_core::Tensor::randn(0f32, 1.0, ${shapeSlice}, &dev)?;`;
  });
  const inputTensors = inputTensorLines.join("\n");

  // Build forward call arguments: &x, &query, &key, ...
  const forwardArgs = paramNames.map(name => `&${name}`).join(", ");

  // Build output shape slice
  const outputShapeSlice = formatShapeSlice(outputShape);

  return `

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn setup() -> (candle_core::Device, VarMap, candle_nn::VarBuilder<'static>) {
        let dev = candle_core::Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        (dev, varmap, vb)
    }

    #[test]
    fn test_forward() -> candle_core::Result<()> {
        let (dev, _varmap, vb) = setup();
        let model = ${className}::new(vb)?;
${inputTensors}
        let output = model.forward(${forwardArgs})?;
        assert_eq!(output.dims(), ${outputShapeSlice});
        Ok(())
    }

    #[test]
    fn test_save_load() -> candle_core::Result<()> {
        let (dev, varmap, vb) = setup();
        let model = ${className}::new(vb)?;
${inputTensors}
        let output_before = model.forward(${forwardArgs})?;

        let dir = std::env::temp_dir().join("mlmd-e2e");
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let path = dir.join("${className}.safetensors");
        let _ = std::fs::remove_file(&path);
        varmap.save(&path)?;

        let mut varmap2 = VarMap::new();
        let vb2 = candle_nn::VarBuilder::from_varmap(&varmap2, candle_core::DType::F32, &dev);
        let model2 = ${className}::new(vb2)?;
        varmap2.load(&path)?;
        let output_after = model2.forward(${forwardArgs})?;

        let diff = (output_before - &output_after)?.abs()?.sum_all()?;
        assert!(diff.to_vec0::<f32>()? < 1e-5);

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}
`;
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
      structFieldLines.push(`    ${candleInit.field},`);

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
    // Wrap in `let` since plugin templates provide `{var} = expr?;` but not the `let` keyword
    const line = result.forward.startsWith("let ") ? result.forward : `let ${result.forward}`;
    forwardLines.push(indentLines(line, "        "));
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
    `use candle_core::{ModuleT, Result, Tensor};`,
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

  // Append generated test module if applicable
  // Prepend batch dimension (1) to all shapes since MLMD shapes don't include batch dim
  const inputShapes: number[][] = [];
  for (const b of sorted) {
    if (b.type === "Input") {
      const shape = b.outputShapes[0] ?? [];
      inputShapes.push([1, ...shape]);  // add batch dim
    }
  }
  // Get output shape from the Output block's inputShapes (shape feeding into the Output node)
  const outputBlock = sorted.find(b => b.type === "Output");
  const outputShapeBase = outputBlock?.inputShapes?.[0] ?? null;
  const outputShape = outputShapeBase ? [1, ...outputShapeBase] : null;  // add batch dim

  const testModule = generateTestModule(className, forwardParams, inputShapes, outputShape);
  const finalContent = testModule ? content + testModule + "\n" : content;

  return [{ path: `${className}.rs`, content: finalContent }];
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

registerTarget({
  name: "candle",
  fileExtension: ".rs",
  generate: generateCandle,
});
