import type { Graph } from '../ast/graph.js';
import type { LayoutResult, LayoutNode, LayoutEdge, Point, GroupLayout } from './layout.js';
import { SvgBuilder } from './svg-builder.js';
import type { ParamValue } from '../ast/nodes.js';

// Block category detection
const MERGE_TYPES = new Set(['Add', 'Mul', 'Sub', 'Div', 'Concat', 'MatMul']);
const ACTIVATION_TYPES = new Set(['ReLU', 'GELU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU', 'Swish', 'Mish', 'Softmax', 'LogSoftmax']);

function blockCategory(type: string): 'merge' | 'activation' | 'default' {
  if (MERGE_TYPES.has(type)) return 'merge';
  if (ACTIVATION_TYPES.has(type)) return 'activation';
  return 'default';
}

function blockColors(type: string): { fill: string; stroke: string } {
  const cat = blockCategory(type);
  if (cat === 'merge') return { fill: '#ede9fe', stroke: '#7c3aed' };
  if (cat === 'activation') return { fill: '#dcfce7', stroke: '#16a34a' };
  return { fill: '#dbeafe', stroke: '#2563eb' };
}

/** Format a ParamValue as a display string */
function formatParamValue(v: ParamValue): string {
  switch (v.kind) {
    case 'number': return String(v.value);
    case 'string': return `"${v.value}"`;
    case 'bool': return v.value ? 'true' : 'false';
    case 'bareword': return v.value;
    case 'shape': return `(${v.dims.join(',')})`;
    case 'list': return `[${v.items.map(formatParamValue).join(',')}]`;
  }
}

/** Build a short param summary like "k=5 f=6" */
function paramSummary(params: Record<string, ParamValue>): string {
  const SHORT: Record<string, string> = {
    filters: 'f', kernel: 'k', kernel_size: 'k', out_features: 'out',
    in_features: 'in', stride: 's', padding: 'p', groups: 'g',
    num_heads: 'h', dropout: 'drop',
  };
  return Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => {
      const key = SHORT[k] ?? k;
      return `${key}=${formatParamValue(v)}`;
    })
    .join(' ');
}

function shapeLabel(shape: number[]): string {
  return `(${shape.join(',')})`;
}

/** Format number with commas, e.g. 62218 → "62,218" */
function formatCount(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

const PADDING = 40;

export function render(graph: Graph, layoutResult: LayoutResult): string {
  const { nodes, edges, groups, width, height } = layoutResult;

  if (nodes.length === 0) {
    const svg = new SvgBuilder(200, 80);
    svg.text(20, 40, 'Empty graph', { 'font-size': '14', fill: '#666' });
    return svg.toString();
  }

  const canvasW = width + PADDING * 2;
  const canvasH = height + PADDING * 2;

  // Account for long-range bezier curves extending past the right or left edge
  const maxCurveOffset = Math.max(0, ...edges.filter(e => e.isLongRange).map(e => e.curveOffset ?? 0));
  const minCurveOffset = Math.min(0, ...edges.filter(e => e.isLongRange).map(e => e.curveOffset ?? 0));
  const leftExtension = Math.abs(Math.min(0, minCurveOffset));
  const adjustedW = canvasW + maxCurveOffset + leftExtension;
  const xOffset = PADDING + leftExtension;

  const svg = new SvgBuilder(adjustedW, canvasH);

  // Build maps for quick lookup
  const nodeById = new Map<string, LayoutNode>(nodes.map(n => [n.block.id, n]));
  const edgeByFrom = new Map<string, LayoutEdge[]>();
  for (const e of edges) {
    const list = edgeByFrom.get(e.from.block.id) ?? [];
    list.push(e);
    edgeByFrom.set(e.from.block.id, list);
  }

  // --- Groups (dashed bounding boxes) ---
  for (const group of groups) {
    const gx = group.x + xOffset;
    const gy = group.y + PADDING;

    svg.rect(gx, gy, group.width, group.height, {
      fill: 'none',
      stroke: '#94a3b8',
      'stroke-width': '1.5',
      'stroke-dasharray': '6,4',
      rx: '8',
    });
    svg.text(gx + 8, gy + 14, group.path.join(' / '), {
      'font-size': '11',
      fill: '#64748b',
      'font-family': 'sans-serif',
      'font-weight': '500',
    });
  }

  // --- Nodes ---
  for (const node of nodes) {
    const { block, x, y, width: w, height: h } = node;
    const px = x + xOffset;
    const py = y + PADDING;
    const cx = px + w / 2;
    const { fill, stroke } = blockColors(block.type);

    // All blocks use rounded rects (no diamonds)
    svg.roundedRect(px, py, w, h, 8, {
      fill,
      stroke,
      'stroke-width': '2',
    });

    // Block type label
    const summary = paramSummary(block.params);
    const label = summary ? `${block.type} ${summary}` : block.type;

    svg.text(cx, py + 14, label, {
      'text-anchor': 'middle',
      'font-size': '11',
      fill: '#1e293b',
      'font-family': 'sans-serif',
      'font-weight': '500',
    });

    // Output shape label(s) (inside block, near bottom) — all outputs for multi-output blocks
    const shapeLabels = block.outputShapes
      .filter(s => s.length > 0)
      .map(s => shapeLabel(s));
    if (shapeLabels.length > 0) {
      svg.text(cx, py + h - 4, shapeLabels.join(' | '), {
        'text-anchor': 'middle',
        'font-size': '8',
        fill: '#64748b',
        'font-family': 'monospace',
      });
    }
  }

  // --- Edges (drawn on top of nodes to avoid vectors hidden behind blocks) ---
  // Find edges with shape info from graph.edges
  const graphEdgeMap = new Map<string, { shape?: number[]; tensorName?: string }>();
  for (const e of graph.edges) {
    graphEdgeMap.set(`${e.from}__${e.to}`, { shape: e.shape, tensorName: e.tensorName });
  }

  for (const edge of edges) {
    const key = `${edge.from.block.id}__${edge.to.block.id}`;
    const graphEdge = graphEdgeMap.get(key);
    const shapeStr = graphEdge?.shape && graphEdge.shape.length > 0 ? shapeLabel(graphEdge.shape) : null;

    // Build the label parts for this edge
    const edgeLabelParts: string[] = [];
    if (edge.isLongRange) {
      edgeLabelParts.push(edge.label ?? '→ skip');
    } else if (graphEdge?.tensorName) {
      edgeLabelParts.push(graphEdge.tensorName);
    }
    if (shapeStr) edgeLabelParts.push(shapeStr);
    const edgeLabel = edgeLabelParts.join(' ');

    if (edge.isLongRange) {
      // Bezier curve using exit/entry ports, supporting left or right side routing
      const p0: Point = edge.exitPort ?? { x: edge.from.x + edge.from.width, y: edge.from.y + edge.from.height / 2 };
      const p3: Point = edge.entryPort ?? { x: edge.to.x + edge.to.width, y: edge.to.y + edge.to.height / 2 };
      const off = edge.curveOffset ?? 80;
      const p1: Point = { x: p0.x + off, y: p0.y };
      const p2: Point = { x: p3.x + off, y: p3.y };
      const d = `M ${p0.x + xOffset} ${p0.y + PADDING} C ${p1.x + xOffset} ${p1.y + PADDING}, ${p2.x + xOffset} ${p2.y + PADDING}, ${p3.x + xOffset} ${p3.y + PADDING}`;
      svg.path(d, {
        stroke: '#9333ea',
        'stroke-width': '1.5',
        fill: 'none',
        'stroke-dasharray': '5,3',
        'marker-end': 'url(#arrowhead)',
      });

      // Label: position near the exit port
      const lx = (edge.labelPosition?.x ?? p0.x + 8) + xOffset;
      const ly = (edge.labelPosition?.y ?? p0.y - 6) + PADDING;
      svg.text(lx, ly, edgeLabel, {
        'font-size': '9',
        fill: '#9333ea',
        'text-anchor': edge.routeSide === 'left' ? 'end' : 'start',
        'font-family': 'monospace',
      });
    } else if (edge.points.length >= 2) {
      // Build polyline path
      const pts = edge.points;
      let d = `M ${pts[0].x + xOffset} ${pts[0].y + PADDING}`;
      for (let i = 1; i < pts.length; i++) {
        d += ` L ${pts[i].x + xOffset} ${pts[i].y + PADDING}`;
      }
      svg.path(d, {
        stroke: '#64748b',
        'stroke-width': '1.5',
        fill: 'none',
        'marker-end': 'url(#arrowhead)',
      });

      // Label centered in vertical gap between blocks (name + shape when available)
      if (edgeLabel) {
        const mid = Math.floor(pts.length / 2);
        const p1 = pts[mid - 1] ?? pts[0];
        const p2 = pts[mid] ?? pts[pts.length - 1];
        const lx = (edge.labelPosition?.x ?? (p1.x + p2.x) / 2 + 4) + xOffset;
        const ly = (edge.labelPosition?.y ?? (p1.y + p2.y) / 2 + 4) + PADDING;
        svg.text(lx, ly, edgeLabel, {
          'font-size': '8',
          fill: '#94a3b8',
          'font-family': 'monospace',
        });
      }
    }
  }

  // --- Total parameter count ---
  let totalParams = 0;
  for (const node of nodes) {
    if (node.block.paramCount !== undefined) totalParams += node.block.paramCount;
  }
  svg.text(canvasW - 10, canvasH - 6, `Total params: ${formatCount(totalParams)}`, {
    'text-anchor': 'end',
    'font-size': '10',
    fill: '#94a3b8',
    'font-family': 'sans-serif',
  });

  return svg.toString();
}
