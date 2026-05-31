import { describe, it, expect } from 'vitest';
import { chooseSide, detectLabelOverlaps, resolveOverlaps, layout } from '../../src/visualize/layout.js';
import type { LayoutNode, LabelRect } from '../../src/visualize/layout.js';
import type { Graph, Block, Edge } from '../../src/ast/graph.js';

function mockNode(id: string, x: number, y: number, w = 60, h = 28): LayoutNode {
  return { block: { id, type: 'Linear', params: {}, inputShapes: [], outputShapes: [], loc: { line: 1, col: 0 } }, x, y, width: w, height: h };
}

function makeLayerMap(nodes: LayoutNode[]): Map<string, number> {
  const map = new Map<string, number>();
  nodes.forEach((n, i) => map.set(n.block.id, i));
  return map;
}

function makeBlock(id: string): Block {
  return { id, type: 'Linear', params: {}, inputShapes: [], outputShapes: [], loc: { line: 1, col: 0 } };
}

function makeGraph(ids: string[], edgePairs: [string, string][]): Graph {
  const blocks = ids.map(makeBlock);
  const edges: Edge[] = edgePairs.map(([from, to]) => ({ from, to }));
  return { blocks, edges, groups: [] };
}

describe('chooseSide', () => {
  it('left-positioned source picks left when intervening nodes are on the right', () => {
    // fromNode at x=0, toNode at x=120; intervening nodes at x=100..200 (on the right)
    const fromNode = mockNode('from', 0, 0);
    const toNode   = mockNode('to',  120, 200);
    const intervening = [
      mockNode('n1', 100, 50),
      mockNode('n2', 140, 80),
      mockNode('n3', 180, 120),
    ];
    const allNodes = [fromNode, toNode, ...intervening];
    const layerMap = new Map([
      ['from', 0], ['to', 4],
      ['n1', 1], ['n2', 2], ['n3', 3],
    ]);
    const result = chooseSide(fromNode, toNode, allNodes, 0, 4, layerMap, 300);
    expect(result).toBe('left');
  });

  it('right-positioned source picks right when intervening nodes are on the left', () => {
    // fromNode at x=200, toNode at x=180; intervening nodes at x=0..60 (on the left)
    const fromNode = mockNode('from', 200, 0);
    const toNode   = mockNode('to',   180, 200);
    const intervening = [
      mockNode('n1', 0,  50),
      mockNode('n2', 20, 80),
      mockNode('n3', 60, 120),
    ];
    const allNodes = [fromNode, toNode, ...intervening];
    const layerMap = new Map([
      ['from', 0], ['to', 4],
      ['n1', 1], ['n2', 2], ['n3', 3],
    ]);
    const result = chooseSide(fromNode, toNode, allNodes, 0, 4, layerMap, 300);
    expect(result).toBe('right');
  });

  it('tiebreak by center: source at x=0 in 300-wide diagram picks left', () => {
    const fromNode = mockNode('from', 0, 0);
    const toNode   = mockNode('to',   0, 200);
    const layerMap = new Map([['from', 0], ['to', 1]]);
    const result = chooseSide(fromNode, toNode, [fromNode, toNode], 0, 1, layerMap, 300);
    expect(result).toBe('left');
  });
});

describe('detectLabelOverlaps', () => {
  it('returns one pair for two overlapping rects', () => {
    const labels: LabelRect[] = [
      { edgeIndex: 0, x: 10, y: 10, width: 50, height: 12 },
      { edgeIndex: 1, x: 30, y: 18, width: 50, height: 12 },
    ];
    expect(detectLabelOverlaps(labels)).toEqual([[0, 1]]);
  });

  it('returns empty array for two non-overlapping rects', () => {
    const labels: LabelRect[] = [
      { edgeIndex: 0, x: 10, y: 10, width: 50, height: 12 },
      { edgeIndex: 1, x: 10, y: 30, width: 50, height: 12 },
    ];
    expect(detectLabelOverlaps(labels)).toHaveLength(0);
  });
});

describe('resolveOverlaps', () => {
  it('separates two colliding labels', () => {
    const labels: LabelRect[] = [
      { edgeIndex: 0, x: 10, y: 10, width: 50, height: 12 },
      { edgeIndex: 1, x: 10, y: 18, width: 50, height: 12 }, // overlaps: y=18 < 10+12=22
    ];
    resolveOverlaps(labels);
    expect(detectLabelOverlaps(labels)).toHaveLength(0);
  });
});

describe('layout – inception-style graph with long-range edge', () => {
  it('long-range edge (span=6) has routeSide set', () => {
    // a→b→c→d→e→f→g (chain), plus long-range a→g (span 6)
    const g = makeGraph(
      ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
      [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e'], ['e', 'f'], ['f', 'g'], ['a', 'g']],
    );
    const result = layout(g);
    const longEdge = result.edges.find(e => e.from.block.id === 'a' && e.to.block.id === 'g');
    expect(longEdge).toBeDefined();
    expect(longEdge!.isLongRange).toBe(true);
    expect(longEdge!.routeSide === 'left' || longEdge!.routeSide === 'right').toBe(true);
  });
});
