import type { Graph, Block } from "../ast/graph.js";
import type { BlockDef } from "../blocks/types.js";
import type { SourceLoc } from "../ast/nodes.js";
import type { LintDiagnostic } from "./index.js";
import { inferShapes } from "../shape/infer.js";
import { topoSortIds } from "../ast/graph-utils.js";

const UNKNOWN_LOC: SourceLoc = { line: 0, col: 0, offset: 0 };

function blockLoc(b: Block): SourceLoc {
  return b.loc ?? UNKNOWN_LOC;
}

/** Rule: cycle detection (must run first) */
export function checkCycles(
  graph: Graph,
  _registry: Map<string, BlockDef>
): LintDiagnostic[] {
  const order = topoSortIds(graph.blocks, graph.edges);
  if (order !== null) return [];

  // Find all blocks involved in a cycle using Kahn inversion
  const ids = graph.blocks.map((b) => b.id);
  const inDegree = new Map<string, number>(ids.map((id) => [id, 0]));
  const adjOut = new Map<string, string[]>(ids.map((id) => [id, []]));

  for (const e of graph.edges) {
    if (!inDegree.has(e.to) || !inDegree.has(e.from)) continue;
    inDegree.set(e.to, (inDegree.get(e.to) ?? 0) + 1);
    adjOut.get(e.from)!.push(e.to);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  const visited: string[] = [];
  while (queue.length > 0) {
    const id = queue.shift()!;
    visited.push(id);
    for (const next of adjOut.get(id) ?? []) {
      const deg = (inDegree.get(next) ?? 0) - 1;
      inDegree.set(next, deg);
      if (deg === 0) queue.push(next);
    }
  }

  const cycleIds = ids.filter((id) => !visited.includes(id));
  const blockMap = new Map(graph.blocks.map((b) => [b.id, b]));
  return cycleIds.map((id) => {
    const b = blockMap.get(id)!;
    return {
      severity: "error" as const,
      message: `Block "${b.type}" (${id}) is part of a cycle`,
      loc: blockLoc(b),
      rule: "cycle",
    };
  });
}

/** Rule: missing plugin/block type (unknown type not in registry) */
export function checkMissingPlugins(
  graph: Graph,
  registry: Map<string, BlockDef>
): LintDiagnostic[] {
  const diags: LintDiagnostic[] = [];
  for (const b of graph.blocks) {
    if (!registry.has(b.type)) {
      diags.push({
        severity: "error",
        message: `Unknown block type: "${b.type}"`,
        loc: blockLoc(b),
        rule: "missing-plugin",
      });
    }
  }
  return diags;
}

/** Rule: missing required params */
export function checkMissingParams(
  graph: Graph,
  registry: Map<string, BlockDef>
): LintDiagnostic[] {
  const diags: LintDiagnostic[] = [];
  for (const b of graph.blocks) {
    const def = registry.get(b.type);
    if (!def) continue;
    for (const spec of def.params) {
      if (spec.required && !(spec.name in b.params)) {
        diags.push({
          severity: "error",
          message: `Block "${b.type}" (${b.id}) is missing required param "${spec.name}"`,
          loc: blockLoc(b),
          rule: "missing-param",
        });
      }
    }
  }
  return diags;
}

/** Rule: shape compatibility (runs inferShapes, collects errors) */
export function checkShapes(
  graph: Graph,
  registry: Map<string, BlockDef>
): LintDiagnostic[] {
  const { errors, graph: inferredGraph } = inferShapes(graph, registry);
  const blockMap = new Map(graph.blocks.map((b) => [b.id, b]));
  const diags: LintDiagnostic[] = [];

  // Collect errors from inferShapes
  for (const e of errors) {
    if (e.message === "Cycle detected in graph") continue; // covered by checkCycles
    const b = e.blockId ? blockMap.get(e.blockId) : undefined;
    diags.push({
      severity: "error",
      message: e.message,
      loc: b ? blockLoc(b) : UNKNOWN_LOC,
      rule: "shape-mismatch",
    });
  }

  // Additional check: elementwise merge blocks require all input shapes to match
  // (inferShapes doesn't validate this, it just returns inputs[0])
  const inferredBlockMap = new Map(inferredGraph.blocks.map((b) => [b.id, b]));
  for (const b of graph.blocks) {
    const def = registry.get(b.type);
    if (!def) continue;
    const inferredBlock = inferredBlockMap.get(b.id);
    if (!inferredBlock) continue;
    const inputShapes = inferredBlock.inputShapes;
    if (inputShapes.length < 2) continue;

    // Check if all input shapes are equal (for elementwise ops)
    // We identify elementwise ops as blocks where the inferShape simply returns
    // inputs[0] unchanged — a heuristic: if the block has no "axis" or dimension
    // params and has multiple inputs, validate shapes match.
    // More precisely: we detect them by trying inferShape with mismatched shapes.
    // Instead, use a simpler convention: check if block type is in the known set,
    // or if any two input shapes differ and the inferred output equals one of the inputs.
    const outShapes = inferredBlock.outputShapes;
    if (outShapes.length === 1) {
      const out = outShapes[0];
      const firstIn = inputShapes[0];
      if (
        out.length === firstIn.length &&
        out.every((d, i) => d === firstIn[i])
      ) {
        // Output matches first input — this may be an elementwise op
        // Verify all inputs match the first
        for (let i = 1; i < inputShapes.length; i++) {
          const inp = inputShapes[i];
          const mismatch =
            inp.length !== firstIn.length ||
            inp.some((d, j) => d !== firstIn[j]);
          if (mismatch) {
            diags.push({
              severity: "error",
              message: `Block "${b.type}" (${b.id}): input shape [${inp.join(",")}] does not match expected [${firstIn.join(",")}]`,
              loc: blockLoc(b),
              rule: "shape-mismatch",
            });
          }
        }
      }
    }
  }

  return diags;
}

/** Rule: duplicate tensor names */
export function checkDuplicateTensorNames(
  graph: Graph,
  _registry: Map<string, BlockDef>
): LintDiagnostic[] {
  const seen = new Map<string, { edge: typeof graph.edges[number]; count: number }>();
  for (const e of graph.edges) {
    if (!e.tensorName) continue;
    if (seen.has(e.tensorName)) {
      seen.get(e.tensorName)!.count++;
    } else {
      seen.set(e.tensorName, { edge: e, count: 1 });
    }
  }

  const diags: LintDiagnostic[] = [];
  for (const [name, { count }] of seen) {
    if (count > 1) {
      diags.push({
        severity: "error",
        message: `Tensor name "${name}" is defined ${count} times`,
        loc: UNKNOWN_LOC,
        rule: "duplicate-tensor-name",
      });
    }
  }
  return diags;
}

/** Rule: undefined tensor references (TensorJoin sources that were never named) */
export function checkUndefinedTensorRefs(
  graph: Graph,
  _registry: Map<string, BlockDef>
): LintDiagnostic[] {
  // Collect all defined tensor names from edges
  const defined = new Set<string>();
  for (const e of graph.edges) {
    if (e.tensorName) defined.add(e.tensorName);
  }

  // Collect all consumed tensor names
  // In the graph IR, tensor joins are represented as edges whose `from` block
  // may have a special convention. However the simplest representation is:
  // edges created by TensorJoin carry no tensorName themselves but the
  // "source" names are stored on the joining block params or via multiple
  // incoming edges. We detect undefined refs by looking for edges whose
  // tensorName is referenced but never produced.
  //
  // The build-graph step encodes TensorJoin sources as incoming edges to
  // the merge/join block. We check if any edge references a tensor name
  // that was never defined.
  // Actually, we look at the "joinSources" metadata if present.
  // Since the graph IR doesn't explicitly store join sources as names,
  // we check params for any "sources" or look at block.params for list refs.
  // The most reliable approach: scan blocks of any type that consume named
  // tensors. We'll rely on the convention that join-type blocks can have a
  // "sources" param that lists tensor names.

  // In lieu of a richer IR, check if any block has a param of kind "list"
  // whose items are bareword or string referencing undefined tensors.
  // This handles TensorJoin patterns.

  const diags: LintDiagnostic[] = [];
  const blockMap = new Map(graph.blocks.map((b) => [b.id, b]));

  // Check join_sources metadata on blocks (stored as _joinSources convention)
  for (const b of graph.blocks) {
    const joinSources = (b as unknown as Record<string, unknown>)["joinSources"];
    if (Array.isArray(joinSources)) {
      for (const src of joinSources as string[]) {
        if (!defined.has(src)) {
          diags.push({
            severity: "error",
            message: `Undefined tensor reference: "${src}"`,
            loc: blockLoc(b),
            rule: "undefined-tensor-ref",
          });
        }
      }
    }
  }

  return diags;
}

/** Rule: unused named tensors (defined but never consumed) */
export function checkUnusedTensors(
  graph: Graph,
  _registry: Map<string, BlockDef>
): LintDiagnostic[] {
  // Collect all defined tensor names
  const defined = new Set<string>();
  for (const e of graph.edges) {
    if (e.tensorName) defined.add(e.tensorName);
  }

  // Collect consumed tensor names from joinSources
  const consumed = new Set<string>();
  for (const b of graph.blocks) {
    const joinSources = (b as unknown as Record<string, unknown>)["joinSources"];
    if (Array.isArray(joinSources)) {
      for (const src of joinSources as string[]) {
        consumed.add(src);
      }
    }
  }

  const diags: LintDiagnostic[] = [];
  // Named tensors that are used as intermediate edges feeding into other blocks
  // are NOT unused — only those where the tensorName edge's `to` block never
  // uses it via joinSources AND the edge has a downstream consumer already
  // tracked. Since named edges still appear as normal edges in the graph, we
  // only flag names that have no consumers at all beyond the tensorName itself.
  //
  // Simplified: a tensor name is "unused" if it was defined (edge has tensorName)
  // and never appears in any block's joinSources, AND the edge's `to` block
  // doesn't exist (dangling).
  for (const name of defined) {
    if (!consumed.has(name)) {
      // Check if the named edge has a real downstream block
      const edge = graph.edges.find((e) => e.tensorName === name);
      if (!edge) continue;
      // If `to` is empty or the target block doesn't exist, it's unused
      const hasDownstream = graph.blocks.some((b) => b.id === edge.to);
      if (!hasDownstream) {
        diags.push({
          severity: "warning",
          message: `Named tensor "${name}" is created but never consumed`,
          loc: UNKNOWN_LOC,
          rule: "unused-tensor",
        });
      }
    }
  }

  return diags;
}
