import { describe, it, expect } from 'vitest';
import { SvgBuilder } from '../../src/visualize/svg-builder.js';

describe('SvgBuilder', () => {
  it('produces a valid SVG document with opening/closing tags', () => {
    const svg = new SvgBuilder();
    const out = svg.toString();
    expect(out).toContain('<svg');
    expect(out).toContain('</svg>');
    expect(out).toContain('xmlns="http://www.w3.org/2000/svg"');
    expect(out).toContain('viewBox');
  });

  it('rect() produces a <rect> element', () => {
    const svg = new SvgBuilder();
    svg.rect(10, 20, 100, 50);
    const out = svg.toString();
    expect(out).toContain('<rect');
    expect(out).toContain('x="10"');
    expect(out).toContain('y="20"');
    expect(out).toContain('width="100"');
    expect(out).toContain('height="50"');
  });

  it('roundedRect() produces a <rect> with rx/ry', () => {
    const svg = new SvgBuilder();
    svg.roundedRect(0, 0, 80, 40, 8);
    const out = svg.toString();
    expect(out).toContain('rx="8"');
    expect(out).toContain('ry="8"');
  });

  it('diamond() produces a <path>', () => {
    const svg = new SvgBuilder();
    svg.diamond(50, 50, 60, 40);
    const out = svg.toString();
    expect(out).toContain('<path');
  });

  it('text() produces a <text> element', () => {
    const svg = new SvgBuilder();
    svg.text(5, 15, 'hello');
    const out = svg.toString();
    expect(out).toContain('<text');
    expect(out).toContain('hello');
  });

  it('path() produces a <path> element', () => {
    const svg = new SvgBuilder();
    svg.path('M 0 0 L 100 100');
    const out = svg.toString();
    expect(out).toContain('<path');
    expect(out).toContain('M 0 0 L 100 100');
  });

  it('arrow() produces a <line> with arrowhead marker', () => {
    const svg = new SvgBuilder();
    svg.arrow(0, 0, 100, 100);
    const out = svg.toString();
    expect(out).toContain('<line');
    expect(out).toContain('marker-end');
    expect(out).toContain('arrowhead');
    expect(out).toContain('<marker');
  });

  it('group() nesting produces nested <g> elements', () => {
    const svg = new SvgBuilder();
    const g = svg.group('layer1');
    g.rect(0, 0, 50, 50);
    const g2 = g.group('layer2', 'translate(10,10)');
    g2.text(0, 0, 'nested');
    const out = svg.toString();
    expect(out).toContain('<g id="layer1"');
    expect(out).toContain('<g id="layer2"');
    expect(out).toContain('translate(10,10)');
    expect(out).toContain('nested');
  });

  it('opts are rendered as attributes', () => {
    const svg = new SvgBuilder();
    svg.rect(0, 0, 10, 10, { fill: 'red', stroke: 'blue' });
    const out = svg.toString();
    expect(out).toContain('fill="red"');
    expect(out).toContain('stroke="blue"');
  });

  it('method chaining works', () => {
    const svg = new SvgBuilder();
    expect(() => svg.rect(0, 0, 10, 10).text(5, 5, 'x').path('M0 0')).not.toThrow();
  });
});
