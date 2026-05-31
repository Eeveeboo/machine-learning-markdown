import { describe, it, expect, beforeEach } from "vitest";
import { TextDocument } from "vscode-languageserver-textdocument";
import { CompletionItemKind } from "vscode-languageserver/node.js";
import { getCompletions } from "../../src/lsp/completions.js";
import { getHover } from "../../src/lsp/hover.js";
import { getDefinition } from "../../src/lsp/definition.js";
import { getReferences } from "../../src/lsp/references.js";
import type { BlockDef } from "../../src/blocks/types.js";

function makeDoc(content: string) {
  return TextDocument.create("file:///test.nnml", "nnml", 1, content);
}

function makeRegistry(defs: BlockDef[]): Map<string, BlockDef> {
  const map = new Map<string, BlockDef>();
  for (const d of defs) map.set(d.name, d);
  return map;
}

const conv2d: BlockDef = {
  name: "Conv2d",
  params: [
    { name: "filters", type: "number", required: true },
    { name: "kernel", type: "shape", required: false },
  ],
  inferShape: () => [],
};

const relu: BlockDef = {
  name: "ReLU",
  params: [],
  inferShape: () => [],
};

const reg = makeRegistry([conv2d, relu]);

describe("getCompletions", () => {
  it("suggests block type names at start of line (partial: Conv)", () => {
    const doc = makeDoc("Conv");
    const items = getCompletions(doc, { line: 0, character: 4 }, reg);
    const labels = items.map((i) => i.label);
    expect(labels).toContain("Conv2d");
  });

  it("suggests block type names after `->`", () => {
    const doc = makeDoc("-> ");
    const items = getCompletions(doc, { line: 0, character: 3 }, reg);
    const labels = items.map((i) => i.label);
    expect(labels).toContain("Conv2d");
    expect(labels).toContain("ReLU");
  });

  it("suggests param names inside block parens", () => {
    const doc = makeDoc("Conv2d(");
    const items = getCompletions(doc, { line: 0, character: 7 }, reg);
    const labels = items.map((i) => i.label);
    expect(labels).toContain("filters");
    expect(labels).toContain("kernel");
  });

  it("suggests tensor names inside [...]", () => {
    const doc = makeDoc("ReLU -> [x]\nConv2d([x,");
    const items = getCompletions(doc, { line: 1, character: 10 }, reg);
    const labels = items.map((i) => i.label);
    expect(labels).toContain("x");
  });

  it("tensor completion kind is Variable", () => {
    const doc = makeDoc("ReLU -> [myTensor]\n[");
    const items = getCompletions(doc, { line: 1, character: 1 }, reg);
    const t = items.find((i) => i.label === "myTensor");
    expect(t).toBeDefined();
    expect(t!.kind).toBe(CompletionItemKind.Variable);
  });
});

describe("getHover", () => {
  it("returns null for unknown word", () => {
    const doc = makeDoc("Unknown");
    expect(getHover(doc, { line: 0, character: 3 }, reg)).toBeNull();
  });

  it("returns markdown hover for known block", () => {
    const doc = makeDoc("Conv2d");
    const hover = getHover(doc, { line: 0, character: 3 }, reg);
    expect(hover).not.toBeNull();
    expect(hover!.contents).toMatchObject({ kind: "markdown" });
    const value = (hover!.contents as { value: string }).value;
    expect(value).toContain("Conv2d");
    expect(value).toContain("filters");
    expect(value).toContain("kernel");
  });

  it("shows required/optional distinction", () => {
    const doc = makeDoc("Conv2d");
    const hover = getHover(doc, { line: 0, character: 0 }, reg);
    const value = (hover!.contents as { value: string }).value;
    // filters is required (no ?), kernel is optional (has ?)
    expect(value).toMatch(/filters: number(?!\?)/);
    expect(value).toContain("kernel: shape?");
  });
});

describe("getDefinition", () => {
  it("returns null when tensor not found", () => {
    const doc = makeDoc("ReLU");
    expect(getDefinition(doc, { line: 0, character: 2 })).toBeNull();
  });

  it("jumps to the declaration line", () => {
    const doc = makeDoc("ReLU -> [myOut]\nConv2d([myOut])");
    // cursor on 'myOut' in line 1
    const loc = getDefinition(doc, { line: 1, character: 10 });
    expect(loc).not.toBeNull();
    expect(loc!.uri).toBe("file:///test.nnml");
    expect(loc!.range.start.line).toBe(0);
    const declLine = "ReLU -> [myOut]";
    const col = declLine.indexOf("myOut");
    expect(loc!.range.start.character).toBe(col);
  });
});

describe("getReferences", () => {
  it("returns empty for unknown tensor", () => {
    const doc = makeDoc("ReLU");
    expect(getReferences(doc, { line: 0, character: 2 })).toEqual([]);
  });

  it("finds declaration and usage", () => {
    const doc = makeDoc("ReLU -> [feat]\nConv2d([feat, feat])");
    // cursor on 'feat' in line 0
    const refs = getReferences(doc, { line: 0, character: 10 });
    expect(refs.length).toBeGreaterThanOrEqual(2);
    const lines = refs.map((r) => r.range.start.line);
    expect(lines).toContain(0); // declaration
    expect(lines).toContain(1); // usage
  });

  it("all references have correct uri", () => {
    const doc = makeDoc("A -> [t]\nB([t])");
    const refs = getReferences(doc, { line: 0, character: 6 });
    for (const r of refs) {
      expect(r.uri).toBe("file:///test.nnml");
    }
  });
});
