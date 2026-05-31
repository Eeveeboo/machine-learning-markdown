import { describe, it, expect } from 'vitest';
import { render } from '../../src/visualize/renderer.js';
import { layout } from '../../src/visualize/layout.js';
import type { Graph, Block, Edge } from '../../src/ast/graph.js';

function makeBlock(id: string, type = 'Linear', params: Record<string, unknown> = {}, outputShapes: number[][] = []): Block {
  return { id, type, params, inputShapes: [], outputShapes, loc: { line: 1, col: 0 } };
}

function makeGraph(blocks: Block[], edgePairs: [string, string][], shapes?: Record<string, number[]>): Graph {
  const edges: Edge[] = edgePairs.map(([from, to]) => ({
    from, to, shape: shapes?.[`${from}->${to}`],
  }));
  return { blocks, edges, groups: [] };
}

describe('render – empty graph', () => {
  it('returns valid SVG with svg tag', () => {
    const g: Graph = { blocks: [], edges: [], groups: [] };
    const result = render(g, layout(g));
    expect(result).toContain('<svg');
    expect(result).toContain('</svg>');
  });
});

describe('render – single block', () => {
  it('produces SVG with rect and text', () => {
    const blocks = [makeBlock('a', 'Conv2d', { filters: 6, kernel: 5 }, [[6, 24, 24]])];
    const g = makeGraph(blocks, []);
    const svg = render(g, layout(g));
    expect(svg).toContain('<svg');
    expect(svg).toContain('<rect');
    expect(svg).toContain('<text');
    expect(svg).toContain('</svg>');
    expect(svg).toContain('Conv2d');
  });
});

describe('render – merge block', () => {
  it('renders Add as rounded rect with purple color', () => {
    const blocks = [makeBlock('a', 'Add')];
    const g = makeGraph(blocks, []);
    const svg = render(g, layout(g));
    expect(svg).toContain('<rect');
    expect(svg).toContain('#ede9fe');
    expect(svg).toContain('#7c3aed');
    expect(svg).toContain('Add');
  });
});

describe('render – activation block', () => {
  it('renders ReLU with green fill', () => {
    const blocks = [makeBlock('a', 'ReLU')];
    const g = makeGraph(blocks, []);
    const svg = render(g, layout(g));
    expect(svg).toContain('#dcfce7');
  });
});

describe('render – edge with shape label', () => {
  it('includes shape string when edge shape is known', () => {
    const blocks = [
      makeBlock('a', 'Conv2d', {}, [[6, 24, 24]]),
      makeBlock('b', 'ReLU', {}, [[6, 24, 24]]),
    ];
    const g = makeGraph(blocks, [['a', 'b']], { 'a->b': [6, 24, 24] });
    const svg = render(g, layout(g));
    expect(svg).toContain('6,24,24');
  });
});

describe('render – chain produces arrows', () => {
  it('SVG has polyline paths for edges', () => {
    const blocks = [
      makeBlock('a', 'Linear', { out_features: 120 }),
      makeBlock('b', 'ReLU'),
      makeBlock('c', 'Linear', { out_features: 10 }),
    ];
    const g = makeGraph(blocks, [['a', 'b'], ['b', 'c']]);
    const svg = render(g, layout(g));
    expect(svg).toContain('arrowhead');
    expect(svg).toContain('Linear');
    expect(svg).toContain('ReLU');
  });
});

describe('render – group bounding box', () => {
  it('renders dashed rect for groups', () => {
    const blocks = [makeBlock('a', 'Conv2d'), makeBlock('b', 'ReLU')];
    const g: Graph = {
      blocks,
      edges: [{ from: 'a', to: 'b' }],
      groups: [{ path: ['MyModel'], blockIds: ['a', 'b'] }],
    };
    const svg = render(g, layout(g));
    expect(svg).toContain('stroke-dasharray');
    expect(svg).toContain('MyModel');
  });
});

describe('render – named edge label', () => {
  it('includes tensor name on non-long-range edges', () => {
    const blockA = makeBlock('a', 'Conv2d', { filters: 32 }, [[32, 64, 64]]);
    const blockB = makeBlock('b', 'ReLU', {}, [[32, 64, 64]]);
    const g: Graph = {
      blocks: [blockA, blockB],
      edges: [{ from: 'a', to: 'b', tensorName: 'enc1', shape: [32, 64, 64] }],
      groups: [],
    };
    const svg = render(g, layout(g));
    expect(svg).toContain('enc1');
    expect(svg).toContain('32,64,64');
  });
});

describe('render – long-range edge', () => {
  it('renders curved bezier arrow with label for long-range edges', () => {
    // Create a long chain so the edge a->f is long-range
    const blocks = [
      makeBlock('a'), makeBlock('b'), makeBlock('c'),
      makeBlock('d'), makeBlock('e'), makeBlock('f'),
    ];
    const g: Graph = {
      blocks,
      edges: [
        { from: 'a', to: 'b' }, { from: 'b', to: 'c' },
        { from: 'c', to: 'd' }, { from: 'd', to: 'e' },
        { from: 'e', to: 'f' },
        { from: 'a', to: 'f', tensorName: 'skip' }, // long-range
      ],
      groups: [],
    };
    const svg = render(g, layout(g));
    // long range edge rendered as bezier arrow with label
    expect(svg).toContain('skip');
    expect(svg).toContain(' C '); // cubic bezier command
  });
});

describe('render – lenet-style chain', () => {
  it('full pipeline produces valid SVG', () => {
    const blockDefs = [
      { id: 'b0', type: 'Input', params: { shape: [1, 28, 28] }, out: [[1, 28, 28]] },
      { id: 'b1', type: 'Conv2d', params: { filters: 6, kernel: 5 }, out: [[6, 24, 24]] },
      { id: 'b2', type: 'ReLU', params: {}, out: [[6, 24, 24]] },
      { id: 'b3', type: 'MaxPool', params: { kernel: 2, stride: 2 }, out: [[6, 12, 12]] },
      { id: 'b4', type: 'Flatten', params: {}, out: [[864]] },
      { id: 'b5', type: 'Linear', params: { out_features: 120 }, out: [[120]] },
    ];
    const blocks = blockDefs.map(d => makeBlock(d.id, d.type, d.params, d.out));
    const edges: [string, string][] = [['b0','b1'],['b1','b2'],['b2','b3'],['b3','b4'],['b4','b5']];
    const g = makeGraph(blocks, edges);
    const svg = render(g, layout(g));

    expect(svg).toContain('<svg');
    expect(svg).toContain('<rect');
    expect(svg).toContain('<text');
    expect(svg).toContain('</svg>');
    expect(svg).toContain('Conv2d');
    expect(svg).toContain('ReLU');
    expect(svg).toContain('Linear');
  });
});
