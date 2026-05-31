import type { Graph, Block, Edge, Shape } from "../ast/graph.js";
import type { BlockDef } from "../blocks/types.js";
import { topoSortIds } from "../ast/graph-utils.js";

export interface ShapeError {
  blockId: string;
  message: string;
}

export interface ShapeResult {
  graph: Graph;
  errors: ShapeError[];
}

/**
 * Infer shapes through the graph in topological order.
 * Returns a new graph (copy) with outputShapes on blocks and shape on edges filled in.
 */
export function inferShapes(
  graph: Graph,
  registry: Map<string, BlockDef>
): ShapeResult {
  const errors: ShapeError[] = [];

  // Deep-copy blocks and edges to avoid mutating the original
  const blocks: Block[] = graph.blocks.map((b) => ({
    ...b,
    inputShapes: [],
    outputShapes: [],
    params: { ...b.params },
  }));
  const edges: Edge[] = graph.edges.map((e) => ({ ...e, shape: undefined }));

  // Cycle detection
  const order = topoSortIds(blocks, edges);
  if (order === null) {
    return {
      graph: { ...graph, blocks, edges },
      errors: [{ blockId: "", message: "Cycle detected in graph" }],
    };
  }

  const blockMap = new Map<string, Block>(blocks.map((b) => [b.id, b]));

  // For each block, gather outgoing edges and incoming edges
  // incoming: to -> list of {from, edgeIdx}
  const incomingEdges = new Map<string, { from: string; edgeIdx: number }[]>();
  for (const b of blocks) incomingEdges.set(b.id, []);
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    if (incomingEdges.has(e.to)) {
      incomingEdges.get(e.to)!.push({ from: e.from, edgeIdx: i });
    }
  }

  // outgoing: from -> list of edgeIdx
  const outgoingEdges = new Map<string, number[]>();
  for (const b of blocks) outgoingEdges.set(b.id, []);
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    if (outgoingEdges.has(e.from)) {
      outgoingEdges.get(e.from)!.push(i);
    }
  }

  for (const blockId of order) {
    const block = blockMap.get(blockId)!;

    // Collect input shapes from incoming edges (in source block output order)
    const incoming = incomingEdges.get(blockId) ?? [];
    const inputShapes: Shape[] = incoming.map(({ from, edgeIdx }) => {
      const srcBlock = blockMap.get(from);
      return edges[edgeIdx].shape ?? srcBlock?.outputShapes?.[0] ?? [];
    });
    block.inputShapes = inputShapes;

    // Lookup block def and infer output shapes
    const def = registry.get(block.type);
    let outputShapes: Shape[];
    if (!def) {
      errors.push({
        blockId: block.id,
        message: `Unknown block type: "${block.type}"`,
      });
      outputShapes = [];
    } else {
      try {
        outputShapes = def.inferShape(inputShapes, block.params);
      } catch (err: unknown) {
        errors.push({
          blockId: block.id,
          message: err instanceof Error ? err.message : String(err),
        });
        outputShapes = [];
      }
    }
    block.outputShapes = outputShapes;

    // Copy visual config from BlockDef
    block.showDepth = def?.showDepth ?? true;

    // Compute parameter count
    if (def?.paramCount) {
      try {
        block.paramCount = def.paramCount(inputShapes, block.params);
      } catch {
        block.paramCount = undefined;
      }
    }

    // Propagate to outgoing edges
    const outIdxs = outgoingEdges.get(blockId) ?? [];
    for (const idx of outIdxs) {
      edges[idx] = { ...edges[idx], shape: outputShapes[0] ?? [] };
    }
  }

  return {
    graph: { ...graph, blocks, edges },
    errors,
  };
}
