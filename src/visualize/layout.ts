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
  routeSide?: 'left' | 'right';
  exitPort?: Point;
  entryPort?: Point;
  labelPosition?: Point;
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
 */
function computeBlockWidth(block: Block): number {
  const shape = block.outputShapes[0];
  if (!shape || shape.length === 0) return 0;
  let W = shape[shape.length - 1];
  if (shape.length >= 2) W *= shape[shape.length - 1];
  else W *= W;
  const w = Math.sqrt(W) * 8;
  return w;
}

/**
 * Compute block height from the channel depth (first dimension) of the output tensor.
 * For 1D shapes, or blocks with showDepth=false (passthrough blocks like activations, merges),
 * returns 0.
 */
function computeBlockHeight(block: Block): number {
  if (block.showDepth === false) return 0;
  const shape = block.outputShapes[0];
  if (!shape || shape.length < 3) return 0;
  const C = shape[0];
  const h = Math.sqrt(C) * 8;
  return h;
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

export function chooseSide(
  fromNode: LayoutNode,
  toNode: LayoutNode,
  allNodes: LayoutNode[],
  fromLayer: number,
  toLayer: number,
  layerMap: Map<string, number>,
  totalWidth: number
): 'left' | 'right' {
  const leftBound = Math.min(fromNode.x, toNode.x);
  const rightBound = Math.max(fromNode.x + fromNode.width, toNode.x + toNode.width);

  let leftCost = 0;
  let rightCost = 0;

  for (const node of allNodes) {
    const layer = layerMap.get(node.block.id);
    if (layer === undefined) continue;
    if (layer <= fromLayer || layer >= toLayer) continue;

    const nx = node.x;
    const nxEnd = node.x + node.width;

    // overlaps left gutter [0, leftBound]
    if (nx < leftBound && nxEnd > 0) leftCost++;
    // overlaps right gutter [rightBound, totalWidth]
    if (nx < totalWidth && nxEnd > rightBound) rightCost++;
  }

  if (leftCost < rightCost) return 'left';
  if (rightCost < leftCost) return 'right';
  const fromCenter = fromNode.x + fromNode.width / 2;
  return fromCenter < totalWidth / 2 ? 'left' : 'right';
}

function computePorts(
  node: LayoutNode,
  isLongRange: boolean,
  routeSide: 'left' | 'right' | undefined,
  isEntry: boolean,
  index = 0,
  count = 1
): Point {
  if (isLongRange && routeSide === 'left') {
    return { x: node.x, y: node.y + node.height / 2 };
  }
  if (isLongRange && routeSide === 'right') {
    return { x: node.x + node.width, y: node.y + node.height / 2 };
  }
  // Distribute across edge (25%..75% of node width) when multiple edges
  const portX = count > 1
    ? node.x + node.width * (0.25 + 0.5 * (index / (count - 1)))
    : node.x + node.width / 2;
  if (isEntry) {
    return { x: portX, y: node.y };
  }
  return { x: portX, y: node.y + node.height };
}

export interface LabelRect {
  edgeIndex: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

const LABEL_H = 12;
const LABEL_CHAR_W = 5;

export function detectLabelOverlaps(labels: LabelRect[]): Array<[number, number]> {
  const overlaps: Array<[number, number]> = [];
  for (let i = 0; i < labels.length; i++) {
    for (let j = i + 1; j < labels.length; j++) {
      const a = labels[i], b = labels[j];
      const overlapX = a.x < b.x + b.width && a.x + a.width > b.x;
      const overlapY = a.y < b.y + b.height && a.y + a.height > b.y;
      if (overlapX && overlapY) overlaps.push([i, j]);
    }
  }
  return overlaps;
}

export function resolveOverlaps(labels: LabelRect[], maxIterations = 3): void {
  for (let iter = 0; iter < maxIterations; iter++) {
    const overlaps = detectLabelOverlaps(labels);
    if (overlaps.length === 0) break;
    for (const [i, j] of overlaps) {
      const a = labels[i], b = labels[j];
      const overlapAmount = (a.y + a.height) - b.y;
      if (overlapAmount > 0) {
        b.y += overlapAmount + 4;
      }
    }
  }
}

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
  // Resacale all widths and heights proportionally to each-other, respecting the MAX_BLOCK_W and MAX_BLOCK_H
  const maxWidth = Math.max(...widthMap.values());
  const maxHeight = Math.max(...heightMap.values());
  // Such that MIN_BLOCK_W and MIN_BLOCK_H are considered as 0
  const scaleW = maxWidth > MIN_BLOCK_W ? MAX_BLOCK_W / (maxWidth - MIN_BLOCK_W) : 1;
  const scaleH = maxHeight > MIN_BLOCK_H ? MAX_BLOCK_H / (maxHeight - MIN_BLOCK_H) : 1;
  for (const id of widthMap.keys()) {
    const w = widthMap.get(id)!;
    const h = heightMap.get(id)!;
    const scaledW = Math.max(MIN_BLOCK_W, MIN_BLOCK_W + (w - MIN_BLOCK_W) * scaleW);
    const scaledH = Math.max(MIN_BLOCK_H, MIN_BLOCK_H + (h - MIN_BLOCK_H) * scaleH);
    widthMap.set(id, scaledW);
    heightMap.set(id, scaledH);
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

  // Pre-compute outgoing/incoming edge ordering for port distribution
  // Sort outgoing edges by target node x (left-to-right) to reduce crossings
  const outEdgeOrder = new Map<string, { edge: typeof edges[number]; index: number; count: number }[]>();
  const inEdgeOrder = new Map<string, { edge: typeof edges[number]; index: number; count: number }[]>();

  for (const e of edges) {
    if (!nodeMap.get(e.from) || !nodeMap.get(e.to)) continue;
    const span = (layerMap.get(e.to) ?? 0) - (layerMap.get(e.from) ?? 0);
    if (span > 2) continue; // skip long-range edges
    if (!outEdgeOrder.has(e.from)) outEdgeOrder.set(e.from, []);
    if (!inEdgeOrder.has(e.to)) inEdgeOrder.set(e.to, []);
    outEdgeOrder.get(e.from)!.push({ edge: e, index: 0, count: 0 });
    inEdgeOrder.get(e.to)!.push({ edge: e, index: 0, count: 0 });
  }

  // Sort outgoing by target x (left target → left port), assign indices
  for (const [, list] of outEdgeOrder) {
    list.sort((a, b) => (nodeMap.get(a.edge.to)?.x ?? 0) - (nodeMap.get(b.edge.to)?.x ?? 0));
    list.forEach((item, i) => { item.index = i; item.count = list.length; });
  }

  // Sort incoming by source x (left source → left port), assign indices
  for (const [, list] of inEdgeOrder) {
    list.sort((a, b) => (nodeMap.get(a.edge.from)?.x ?? 0) - (nodeMap.get(b.edge.from)?.x ?? 0));
    list.forEach((item, i) => { item.index = i; item.count = list.length; });
  }

  // Build lookup: "from__to" → { outIndex, outCount, inIndex, inCount }
  const edgePortInfo = new Map<string, { outIndex: number; outCount: number; inIndex: number; inCount: number }>();
  for (const [, list] of outEdgeOrder) {
    for (const item of list) {
      const key = `${item.edge.from}__${item.edge.to}`;
      const existing = edgePortInfo.get(key) ?? { outIndex: 0, outCount: 1, inIndex: 0, inCount: 1 };
      edgePortInfo.set(key, { ...existing, outIndex: item.index, outCount: item.count });
    }
  }
  for (const [, list] of inEdgeOrder) {
    for (const item of list) {
      const key = `${item.edge.from}__${item.edge.to}`;
      const existing = edgePortInfo.get(key) ?? { outIndex: 0, outCount: 1, inIndex: 0, inCount: 1 };
      edgePortInfo.set(key, { ...existing, inIndex: item.index, inCount: item.count });
    }
  }

  // Build edges (after group layout so node positions are final)
  const layoutEdges: LayoutEdge[] = [];
  let leftSkipIndex = 0;
  let rightSkipIndex = 0;
  for (const e of edges) {
    const fromNode = nodeMap.get(e.from);
    const toNode = nodeMap.get(e.to);
    if (!fromNode || !toNode) continue;

    const fromLayer = layerMap.get(e.from)!;
    const toLayer = layerMap.get(e.to)!;
    const span = toLayer - fromLayer;
    const isLongRange = span > 2;

    if (isLongRange) {
      const routeSide = chooseSide(fromNode, toNode, [...nodeMap.values()], fromLayer, toLayer, layerMap, totalWidth);
      const exitPort = computePorts(fromNode, true, routeSide, false);
      const entryPort = computePorts(toNode, true, routeSide, true);
      const label = e.tensorName ? `→ ${e.tensorName}` : '→ skip';
      const baseOffset = totalWidth * 0.2 + 60;
      const sideIndex = routeSide === 'left' ? leftSkipIndex++ : rightSkipIndex++;
      const curveOffset = (baseOffset + sideIndex * 40) * (routeSide === 'left' ? -1 : 1);
      layoutEdges.push({ from: fromNode, to: toNode, points: [], isLongRange: true, label, curveOffset, routeSide, exitPort, entryPort });
    } else {
      const portKey = `${e.from}__${e.to}`;
      const portInfo = edgePortInfo.get(portKey);
      const outIndex = portInfo?.outIndex ?? 0;
      const outCount = portInfo?.outCount ?? 1;
      const inIndex = portInfo?.inIndex ?? 0;
      const inCount = portInfo?.inCount ?? 1;
      const exitPort = computePorts(fromNode, false, undefined, false, outIndex, outCount);
      const entryPort = computePorts(toNode, false, undefined, true, inIndex, inCount);
      const fx = exitPort.x;
      const fy = exitPort.y;
      const tx = entryPort.x;
      const ty = entryPort.y;

      let points: Point[];
      if (Math.abs(fx - tx) < 1) {
        points = [{ x: fx, y: fy }, { x: tx, y: ty }];
      } else {
        const MIDPOINT_STAGGER = 10;
        const baseMid = (fy + ty) / 2;
        const mid = inCount > 1
          ? baseMid + (inIndex - (inCount - 1) / 2) * MIDPOINT_STAGGER
          : baseMid;
        points = [
          { x: fx, y: fy },
          { x: fx, y: mid },
          { x: tx, y: mid },
          { x: tx, y: ty },
        ];
      }
      layoutEdges.push({ from: fromNode, to: toNode, points, isLongRange: false, exitPort, entryPort });
    }
  }

  // Compute initial label positions, resolve overlaps, store on edges
  const labelRects: LabelRect[] = [];
  for (let i = 0; i < layoutEdges.length; i++) {
    const edge = layoutEdges[i];
    let lx: number, ly: number;
    if (edge.isLongRange) {
      const p0x = edge.exitPort?.x ?? edge.from.x + edge.from.width;
      if (edge.routeSide === 'left') {
        lx = p0x - 60;
      } else {
        lx = p0x + 8;
      }
      const p0y = edge.exitPort?.y ?? (edge.from.y + edge.from.height / 2);
      ly = p0y - 12;
    } else {
      const pts = edge.points;
      if (pts.length >= 2) {
        const mid = Math.floor(pts.length / 2);
        const p1 = pts[mid - 1] ?? pts[0];
        const p2 = pts[mid] ?? pts[pts.length - 1];
        lx = (p1.x + p2.x) / 2 + 4;
        ly = (p1.y + p2.y) / 2 + 4;
      } else {
        lx = edge.from.x;
        ly = edge.from.y;
      }
    }
    const labelText = edge.label ?? '';
    const labelWidth = labelText.length * LABEL_CHAR_W + 20;
    labelRects.push({ edgeIndex: i, x: lx, y: ly, width: labelWidth, height: LABEL_H });
  }

  resolveOverlaps(labelRects);

  for (const rect of labelRects) {
    layoutEdges[rect.edgeIndex].labelPosition = { x: rect.x, y: rect.y };
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
