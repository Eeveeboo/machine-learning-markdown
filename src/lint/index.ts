import type { Graph } from "../ast/graph.js";
import type { BlockDef } from "../blocks/types.js";
import type { SourceLoc } from "../ast/nodes.js";
import {
  checkCycles,
  checkMissingPlugins,
  checkMissingParams,
  checkShapes,
  checkDuplicateTensorNames,
  checkUndefinedTensorRefs,
  checkUnusedTensors,
} from "./rules.js";

export interface LintDiagnostic {
  severity: "error" | "warning";
  message: string;
  loc: SourceLoc;
  rule: string;
}

/**
 * Run all lint rules against the graph.
 * Cycle detection runs first; shape checks are skipped if cycles are present.
 */
export function lint(
  graph: Graph,
  registry: Map<string, BlockDef>
): LintDiagnostic[] {
  const diags: LintDiagnostic[] = [];

  // 1. Cycle detection must run first
  const cycleErrors = checkCycles(graph, registry);
  diags.push(...cycleErrors);

  if (cycleErrors.length > 0) {
    // Skip shape inference and other order-dependent checks if cycles exist
    diags.push(...checkMissingPlugins(graph, registry));
    diags.push(...checkMissingParams(graph, registry));
    diags.push(...checkDuplicateTensorNames(graph, registry));
    diags.push(...checkUndefinedTensorRefs(graph, registry));
    diags.push(...checkUnusedTensors(graph, registry));
    return diags;
  }

  // 2. Missing plugins
  diags.push(...checkMissingPlugins(graph, registry));

  // 3. Missing required params
  diags.push(...checkMissingParams(graph, registry));

  // 4. Shape compatibility (runs inferShapes internally)
  diags.push(...checkShapes(graph, registry));

  // 5. Duplicate tensor names
  diags.push(...checkDuplicateTensorNames(graph, registry));

  // 6. Undefined tensor references
  diags.push(...checkUndefinedTensorRefs(graph, registry));

  // 7. Unused named tensors
  diags.push(...checkUnusedTensors(graph, registry));

  return diags;
}
