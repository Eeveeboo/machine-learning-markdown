/**
 * Low-level SVG string builder — no DOM dependency.
 */

function escapeXml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function attrs(opts: Record<string, string> = {}): string {
  return Object.entries(opts)
    .map(([k, v]) => ` ${k}="${escapeXml(v)}"`)
    .join('');
}

const ARROWHEAD_MARKER = `<defs>
  <marker id="arrowhead" markerWidth="7" markerHeight="5" refX="7" refY="2.5" markerUnits="userSpaceOnUse" orient="auto">
    <polygon points="0 0, 7 2.5, 0 5" fill="#94a3b8" />
  </marker>
</defs>`;

export class SvgBuilder {
  private _elements: string[] = [];
  private _children: SvgBuilder[] = [];
  private _groupId: string | undefined;
  private _transform: string | undefined;
  private _isRoot: boolean;
  private _width: number;
  private _height: number;

  constructor(
    width = 800,
    height = 600,
    options: { isRoot?: boolean; groupId?: string; transform?: string } = {}
  ) {
    this._width = width;
    this._height = height;
    this._isRoot = options.isRoot ?? true;
    this._groupId = options.groupId;
    this._transform = options.transform;
  }

  rect(x: number, y: number, w: number, h: number, opts?: Record<string, string>): this {
    this._elements.push(
      `<rect x="${x}" y="${y}" width="${w}" height="${h}"${attrs(opts)} />`
    );
    return this;
  }

  roundedRect(
    x: number,
    y: number,
    w: number,
    h: number,
    r: number,
    opts?: Record<string, string>
  ): this {
    this._elements.push(
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" ry="${r}"${attrs(opts)} />`
    );
    return this;
  }

  diamond(cx: number, cy: number, w: number, h: number, opts?: Record<string, string>): this {
    const hw = w / 2;
    const hh = h / 2;
    const d = `M ${cx} ${cy - hh} L ${cx + hw} ${cy} L ${cx} ${cy + hh} L ${cx - hw} ${cy} Z`;
    this._elements.push(`<path d="${d}"${attrs(opts)} />`);
    return this;
  }

  text(x: number, y: number, content: string, opts?: Record<string, string>): this {
    this._elements.push(`<text x="${x}" y="${y}"${attrs(opts)}>${escapeXml(content)}</text>`);
    return this;
  }

  path(d: string, opts?: Record<string, string>): this {
    this._elements.push(`<path d="${d}"${attrs(opts)} />`);
    return this;
  }

  arrow(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    opts?: Record<string, string>
  ): this {
    const merged: Record<string, string> = {
      stroke: '#333',
      'stroke-width': '2',
      fill: 'none',
      'marker-end': 'url(#arrowhead)',
      ...opts,
    };
    this._elements.push(
      `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"${attrs(merged)} />`
    );
    return this;
  }

  group(id: string, transform?: string): SvgBuilder {
    const child = new SvgBuilder(this._width, this._height, {
      isRoot: false,
      groupId: id,
      transform,
    });
    this._children.push(child);
    // Also push a placeholder that we replace in _render
    this._elements.push(`__child_${this._children.length - 1}__`);
    return child;
  }

  private _render(): string {
    let childIdx = 0;
    const lines: string[] = [];
    for (const el of this._elements) {
      const match = el.match(/^__child_(\d+)__$/);
      if (match) {
        const idx = parseInt(match[1], 10);
        lines.push(this._children[idx]._renderGroup());
      } else {
        lines.push(el);
      }
    }
    return lines.join('\n');
  }

  private _renderGroup(): string {
    const transformAttr = this._transform ? ` transform="${escapeXml(this._transform)}"` : '';
    const inner = this._render();
    return `<g id="${escapeXml(this._groupId ?? '')}"${transformAttr}>\n${inner}\n</g>`;
  }

  toString(): string {
    const inner = this._render();
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this._width} ${this._height}" width="${this._width}" height="${this._height}">
${ARROWHEAD_MARKER}
${inner}
</svg>`;
  }
}
