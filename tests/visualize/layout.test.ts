import { describe, it, expect } from 'vitest';
import { layout } from '../../src/visualize/layout.js';
import type { Graph, Block, Edge } from '../../src/ast/graph.js';

function makeBlock(id: string): Block {
  return { id, type: 'Linear', params: {}, inputShapes: [], outputShapes: [], loc: { line: 1, col: 0 } };
}

function makeGraph(ids: string[], edgePairs: [string, string][], tensorNames?: Record<string, string>): Graph {
  const blocks = ids.map(makeBlock);
  const edges: Edge[] = edgePairs.map(([from, to]) => ({
    from, to, tensorName: tensorNames?.[`${from}->${to}`],
  }));
  return { blocks, edges, groups: [] };
}

describe('layout – LeNet (linear chain)', () => {
  const g = makeGraph(
    ['a', 'b', 'c', 'd'],
    [['a', 'b'], ['b', 'c'], ['c', 'd']],
  );
  const result = layout(g);

  it('returns 4 nodes', () => expect(result.nodes).toHaveLength(4));
  it('all nodes in a single column (same x)', () => {
    const xs = result.nodes.map(n => n.x);
    expect(new Set(xs).size).toBe(1);
  });
  it('nodes are vertically ordered', () => {
    const byId = new Map(result.nodes.map(n => [n.block.id, n]));
    expect(byId.get('a')!.y).toBeLessThan(byId.get('b')!.y);
    expect(byId.get('b')!.y).toBeLessThan(byId.get('c')!.y);
    expect(byId.get('c')!.y).toBeLessThan(byId.get('d')!.y);
  });
  it('edges are not long-range', () => {
    expect(result.edges.every(e => !e.isLongRange)).toBe(true);
  });
  it('straight edges have 2 points', () => {
    expect(result.edges.every(e => e.points.length === 2)).toBe(true);
  });
});

describe('layout – ResNet fork (parallel columns)', () => {
  // a -> b, a -> c, b -> d, c -> d  (b and c are parallel)
  const g = makeGraph(
    ['a', 'b', 'c', 'd'],
    [['a', 'b'], ['a', 'c'], ['b', 'd'], ['c', 'd']],
  );
  const result = layout(g);

  it('b and c are at the same layer (same y)', () => {
    const byId = new Map(result.nodes.map(n => [n.block.id, n]));
    expect(byId.get('b')!.y).toBe(byId.get('c')!.y);
  });
  it('b and c are side-by-side (different x)', () => {
    const byId = new Map(result.nodes.map(n => [n.block.id, n]));
    expect(byId.get('b')!.x).not.toBe(byId.get('c')!.x);
  });
  it('a and d are in a different layer from b/c', () => {
    const byId = new Map(result.nodes.map(n => [n.block.id, n]));
    expect(byId.get('a')!.y).toBeLessThan(byId.get('b')!.y);
    expect(byId.get('d')!.y).toBeGreaterThan(byId.get('b')!.y);
  });
});

describe('layout – U-Net skip (long-range edge)', () => {
  // a -> b -> c -> d -> e, plus skip a -> d (span 3)
  const g = makeGraph(
    ['a', 'b', 'c', 'd', 'e'],
    [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e'], ['a', 'd']],
    { 'a->d': 'skip1' },
  );
  const result = layout(g);

  it('skip edge is marked long-range', () => {
    const skip = result.edges.find(e => e.from.block.id === 'a' && e.to.block.id === 'd');
    expect(skip).toBeDefined();
    expect(skip!.isLongRange).toBe(true);
  });
  it('skip edge has a label', () => {
    const skip = result.edges.find(e => e.from.block.id === 'a' && e.to.block.id === 'd');
    expect(skip!.label).toBe('→ skip1');
  });
  it('skip edge has no waypoints', () => {
    const skip = result.edges.find(e => e.from.block.id === 'a' && e.to.block.id === 'd');
    expect(skip!.points).toHaveLength(0);
  });
  it('normal edges are not long-range', () => {
    const normal = result.edges.filter(e => !(e.from.block.id === 'a' && e.to.block.id === 'd'));
    expect(normal.every(e => !e.isLongRange)).toBe(true);
  });
});

describe('layout – empty graph', () => {
  it('handles empty graph gracefully', () => {
    const result = layout({ blocks: [], edges: [], groups: [] });
    expect(result.nodes).toHaveLength(0);
    expect(result.edges).toHaveLength(0);
    expect(result.width).toBe(0);
    expect(result.height).toBe(0);
  });
});

describe('layout – dimensions', () => {
  it('single node has correct dimensions', () => {
    const g = makeGraph(['a'], []);
    const result = layout(g);
    // Empty outputShapes → min width (60), min height (28), result height = 28 + V_GAP = 52
    expect(result.nodes[0].width).toBe(60);
    expect(result.nodes[0].height).toBe(28);
    expect(result.width).toBe(60);
    expect(result.height).toBe(52);
  });
});
