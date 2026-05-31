import type { Graph, Block } from '../ast/graph.js';

export interface Point { x: number; y: number; }

export interface LayoutNode {
  block: Block;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LayoutEdge {
  from: LayoutNode;
  to: LayoutNode;
  points: Point[];
  isLongRange: boolean;
  label?: string;
  /** Rightwards offset for the bezier curve (long-range edges only) */
  curveOffset?: number;
}

export interface GroupLayout {
  path: string[];
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LayoutResult {
  nodes: LayoutNode[];
  edges: LayoutEdge[];
  groups: GroupLayout[];
  width: number;
  height: number;
}

const MIN_BLOCK_W = 60;
const MAX_BLOCK_W = 300;
const MIN_BLOCK_H = 28;
const MAX_BLOCK_H = 200;
const V_GAP = 24;
const H_GAP = 10; // horizontal gap between blocks in same layer

/**
 * Compute block width from the spatial width (last dimension) of the output tensor.
 * width = sqrt(W) × 14 + 60, clamped to [60, 300]
 */
function computeBlockWidth(block: Block): number {
  const shape = block.outputShapes[0];
  if (!shape || shape.length === 0) return MIN_BLOCK_W;
  const W = shape[shape.length - 1];
  const w = Math.round(Math.sqrt(W) * 14 + 60);
  return Math.max(MIN_BLOCK_W, Math.min(MAX_BLOCK_W, w));
}

/**
 * Compute block height from the channel depth (first dimension) of the output tensor.
 * height = sqrt(C) × 8 + 28, clamped to [28, 200]
 * For 1D shapes, or blocks with showDepth=false (passthrough blocks like activations, merges),
 * returns the minimum height.
 */
function computeBlockHeight(block: Block): number {
  if (block.showDepth === false) return MIN_BLOCK_H;
  const shape = block.outputShapes[0];
  if (!shape || shape.length <= 1) return MIN_BLOCK_H;
  const C = shape[0];
  const h = Math.round(Math.sqrt(C) * 8 + 28);
  return Math.max(MIN_BLOCK_H, Math.min(MAX_BLOCK_H, h));
}

interface GroupBuilder {
  path: string[];
  blockIds: string[];
  /** Nodes belonging to this group (resolved after first pass) */
  nodes: LayoutNode[];
}

/** Compute group bounding boxes and ensure no overlap between adjacent groups */
function computeGroupLayouts(graph: Graph, nodeMap: Map<string, LayoutNode>): GroupLayout[] {
  const sidePad = 10;
  const topPad = 28; // room for group title text (font-size 11 + margin)
  const botPad = 10;
  const minGap = 6;

  const builders: GroupBuilder[] = [];
  for (const group of graph.groups) {
    const gNodes = group.blockIds.map(id => nodeMap.get(id)).filter(Boolean) as LayoutNode[];
    if (gNodes.length === 0) continue;
    builders.push({ path: group.path, blockIds: group.blockIds, nodes: gNodes });
  }

  // Sort by Y position of topmost node
  builders.sort((a, b) => Math.min(...a.nodes.map(n => n.y)) - Math.min(...b.nodes.map(n => n.y)));

  const groups: GroupLayout[] = [];

  for (const b of builders) {
    const minX = Math.min(...b.nodes.map(n => n.x)) - sidePad;
    const minY = Math.min(...b.nodes.map(n => n.y)) - topPad;
    const maxX = Math.max(...b.nodes.map(n => n.x + n.width)) + sidePad;
    const maxY = Math.max(...b.nodes.map(n => n.y + n.height)) + botPad;

    let x = minX;
    let y = minY;
    const width = maxX - minX;
    let height = maxY - minY;

    // Resolve vertical overlap with previous groups
    if (groups.length > 0) {
      const prev = groups[groups.length - 1];
      const prevBottom = prev.y + prev.height;
      if (y < prevBottom + minGap) {
        const shift = prevBottom + minGap - y;
        y += shift;
        // Shift member nodes to keep them aligned with the group box
        for (const node of b.nodes) {
          node.y += shift;
          height = Math.max(height, node.y + node.height + botPad - y);
        }
      }
    }
    groups.push({ path: b.path, x, y, width, height });
  }

  return groups;
}

// ---- helpers (unchanged) ----

/** Topological sort — returns block ids in order */
function topoSort(blocks: Block[], edges: { from: string; to: string }[]): string[] {
  const inDeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const b of blocks) { inDeg.set(b.id, 0); adj.set(b.id, []); }
  for (const e of edges) {
    adj.get(e.from)!.push(e.to);
    inDeg.set(e.to, (inDeg.get(e.to) ?? 0) + 1);
  }
  const queue: string[] = [];
  for (const [id, deg] of inDeg) if (deg === 0) queue.push(id);
  const result: string[] = [];
  while (queue.length) {
    const cur = queue.shift()!;
    result.push(cur);
    for (const nxt of adj.get(cur) ?? []) {
      const d = inDeg.get(nxt)! - 1;
      inDeg.set(nxt, d);
      if (d === 0) queue.push(nxt);
    }
  }
  return result;
}

/** Assign each block to a layer (longest path from sources) */
function assignLayers(blockIds: string[], edges: { from: string; to: string }[]): Map<string, number> {
  const layer = new Map<string, number>();
  for (const id of blockIds) layer.set(id, 0);
  for (const id of blockIds) {
    const cur = layer.get(id)!;
    for (const e of edges) {
      if (e.from === id) {
        const dst = e.to;
        if (layer.get(dst)! < cur + 1) layer.set(dst, cur + 1);
      }
    }
  }
  return layer;
}

export function layout(graph: Graph): LayoutResult {
  const { blocks, edges } = graph;
  if (blocks.length === 0) return { nodes: [], edges: [], groups: [], width: 0, height: 0 };

  const sorted = topoSort(blocks, edges);
  const layerMap = assignLayers(sorted, edges);

  // Group blocks by layer
  const byLayer = new Map<number, string[]>();
  for (const id of sorted) {
    const l = layerMap.get(id)!;
    if (!byLayer.has(l)) byLayer.set(l, []);
    byLayer.get(l)!.push(id);
  }

  const sortedLayers = [...byLayer.keys()].sort((a, b) => a - b);
  const blockById = new Map<string, Block>(blocks.map(b => [b.id, b]));
  const nodeMap = new Map<string, LayoutNode>();

  // Pre-compute widths and heights for all blocks
  const widthMap = new Map<string, number>();
  const heightMap = new Map<string, number>();
  for (const b of blocks) {
    widthMap.set(b.id, computeBlockWidth(b));
    heightMap.set(b.id, computeBlockHeight(b));
  }

  // Compute max height per layer
  const layerHeight = new Map<number, number>();
  for (const [layerIdx, ids] of byLayer) {
    const maxH = Math.max(...ids.map(id => heightMap.get(id)!));
    layerHeight.set(layerIdx, maxH);
  }

  // Compute Y positions with variable-height stacking
  const layerY = new Map<number, number>();
  let currentY = 0;
  for (const layerIdx of sortedLayers) {
    layerY.set(layerIdx, currentY);
    currentY += layerHeight.get(layerIdx)! + V_GAP;
  }

  // Find the maximum row width to center everything
  const allRowWidths = sortedLayers.map(l => {
    const ids2 = byLayer.get(l)!;
    const w2 = ids2.map(id => widthMap.get(id)!);
    return w2.reduce((a, b) => a + b, 0) + (ids2.length - 1) * H_GAP;
  });
  const maxRowWidth = Math.max(...allRowWidths);

  // Layout: center each row horizontally, using per-block widths
  for (const layerIdx of sortedLayers) {
    const ids = byLayer.get(layerIdx)!;
    const widths = ids.map(id => widthMap.get(id)!);
    const totalRowWidth = widths.reduce((a, b) => a + b, 0) + (ids.length - 1) * H_GAP;
    const y = layerY.get(layerIdx)!;
    const h = layerHeight.get(layerIdx)!;

    let cursorX = (maxRowWidth - totalRowWidth) / 2;
    ids.forEach((id) => {
      const w = widthMap.get(id)!;
      const bh = heightMap.get(id)!;
      // Center vertically within the layer's max height
      const by = y + (h - bh) / 2;
      nodeMap.set(id, { block: blockById.get(id)!, x: cursorX, y: by, width: w, height: bh });
      cursorX += w + H_GAP;
    });
  }

  const layoutResultHeight = currentY;

  // Compute overall width
  const allWidths = sortedLayers.map(l => {
    const ids = byLayer.get(l)!;
    const w = ids.map(id => widthMap.get(id)!);
    return w.reduce((a, b) => a + b, 0) + (ids.length - 1) * H_GAP;
  });
  const totalWidth = Math.max(...allWidths);

  // Compute group bounds — may shift member nodes to resolve overlap
  const groupLayouts = computeGroupLayouts(graph, nodeMap);

  // Build edges (after group layout so node positions are final)
  const layoutEdges: LayoutEdge[] = [];
  let skipIndex = 0;
  for (const e of edges) {
    const fromNode = nodeMap.get(e.from);
    const toNode = nodeMap.get(e.to);
    if (!fromNode || !toNode) continue;

    const fromLayer = layerMap.get(e.from)!;
    const toLayer = layerMap.get(e.to)!;
    const span = toLayer - fromLayer;
    const isLongRange = span > 2;

    if (isLongRange) {
      const label = e.tensorName ? `→ ${e.tensorName}` : '→ skip';
      const baseOffset = totalWidth * 0.2 + 60;
      const curveOffset = baseOffset + skipIndex * 40;
      skipIndex++;
      layoutEdges.push({ from: fromNode, to: toNode, points: [], isLongRange: true, label, curveOffset });
    } else {
      // Orthogonal L-shaped or straight path
      const fx = fromNode.x + fromNode.width / 2;
      const fy = fromNode.y + fromNode.height;
      const tx = toNode.x + toNode.width / 2;
      const ty = toNode.y;

      let points: Point[];
      if (Math.abs(fx - tx) < 1) {
        // Straight vertical
        points = [{ x: fx, y: fy }, { x: tx, y: ty }];
      } else {
        // L-shaped: go down to midpoint then across
        const mid = (fy + ty) / 2;
        points = [
          { x: fx, y: fy },
          { x: fx, y: mid },
          { x: tx, y: mid },
          { x: tx, y: ty },
        ];
      }
      layoutEdges.push({ from: fromNode, to: toNode, points, isLongRange: false });
    }
  }

  // If groups exist, adjust height to include group vertical space.
  // (computeGroupLayouts may have pushed groups down to avoid overlap.)
  let adjustedHeight = layoutResultHeight;
  if (groupLayouts.length > 0) {
    const maxGroupBottom = Math.max(...groupLayouts.map(g => g.y + g.height));
    adjustedHeight = Math.max(adjustedHeight, maxGroupBottom);
  }

  return {
    nodes: [...nodeMap.values()],
    edges: layoutEdges,
    groups: groupLayouts,
    width: totalWidth,
    height: adjustedHeight,
  };
}
