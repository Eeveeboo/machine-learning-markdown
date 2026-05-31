import type { Graph, Block, Edge } from "./graph.js";

// ---------------------------------------------------------------------------
// DFS-based topological sort (assumes DAG, no cycle detection)
// Returns blocks in topological order (inputs before outputs).
// ---------------------------------------------------------------------------

export function topoSort(graph: Graph): Block[] {
  const adj = buildAdjacency(graph);

  function visit(id: string): void {
    if (visited.has(id)) return;
    visited.add(id);
    const info = adj.get(id);
    if (!info) return;
    for (const inp of info.inputs) visit(inp);
    result.push(info.block);
  }

  const visited = new Set<string>();
  const result: Block[] = [];
  for (const b of graph.blocks) visit(b.id);
  return result;
}

// ---------------------------------------------------------------------------
// Kahn's algorithm topological sort with cycle detection
// Returns sorted block IDs, or null if a cycle is detected.
// ---------------------------------------------------------------------------

export function topoSortIds(
  blocks: Block[],
  edges: Edge[]
): string[] | null {
  const ids = blocks.map((b) => b.id);
  const inDegree = new Map<string, number>(ids.map((id) => [id, 0]));
  const adjOut = new Map<string, string[]>(ids.map((id) => [id, []]));

  for (const e of edges) {
    if (!inDegree.has(e.to) || !inDegree.has(e.from)) continue;
    inDegree.set(e.to, (inDegree.get(e.to) ?? 0) + 1);
    adjOut.get(e.from)!.push(e.to);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  const result: string[] = [];
  while (queue.length > 0) {
    const id = queue.shift()!;
    result.push(id);
    for (const next of adjOut.get(id) ?? []) {
      const deg = (inDegree.get(next) ?? 0) - 1;
      inDegree.set(next, deg);
      if (deg === 0) queue.push(next);
    }
  }

  return result.length === ids.length ? result : null;
}

// ---------------------------------------------------------------------------
// Adjacency helpers
// ---------------------------------------------------------------------------

export interface NodeInfo {
  block: Block;
  inputs: string[];  // block ids that feed into this block
  outputs: string[]; // block ids this block feeds into
}

export function buildAdjacency(graph: Graph): Map<string, NodeInfo> {
  const map = new Map<string, NodeInfo>();
  for (const b of graph.blocks) {
    map.set(b.id, { block: b, inputs: [], outputs: [] });
  }
  for (const e of graph.edges) {
    const from = map.get(e.from);
    const to = map.get(e.to);
    if (from) from.outputs.push(e.to);
    if (to) to.inputs.push(e.from);
  }
  return map;
}
