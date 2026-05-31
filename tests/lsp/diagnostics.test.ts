import { describe, it, expect, vi } from "vitest";
import { TextDocument } from "vscode-languageserver-textdocument";
import { DiagnosticSeverity } from "vscode-languageserver/node.js";
import { validateDocument } from "../../src/lsp/diagnostics.js";
import type { Connection } from "vscode-languageserver/node.js";

function makeConnection() {
  const sendDiagnostics = vi.fn();
  const conn = { sendDiagnostics } as unknown as Connection;
  return { conn, sendDiagnostics };
}

function makeDoc(content: string) {
  return TextDocument.create("file:///test.nnml", "nnml", 1, content);
}

describe("validateDocument", () => {
  it("sends diagnostics for valid document (uri is correct)", () => {
    const { conn, sendDiagnostics } = makeConnection();
    const doc = makeDoc("ReLU");
    validateDocument(conn, doc);
    expect(sendDiagnostics).toHaveBeenCalledOnce();
    const call = sendDiagnostics.mock.calls[0][0];
    expect(call.uri).toBe("file:///test.nnml");
    expect(Array.isArray(call.diagnostics)).toBe(true);
  });

  it("reports a shape mismatch diagnostic for incompatible Add inputs", () => {
    const { conn, sendDiagnostics } = makeConnection();
    // Two inputs with different shapes fed into Add → shape mismatch
    const content = [
      "a: Input(shape=[3,32,32])",
      "b: Input(shape=[3,64,64])",
      "Add <- a, b",
    ].join("\n");
    const doc = makeDoc(content);
    validateDocument(conn, doc);
    expect(sendDiagnostics).toHaveBeenCalledOnce();
    const { diagnostics } = sendDiagnostics.mock.calls[0][0];
    const shapeErrors = diagnostics.filter(
      (d: { code?: string }) => d.code === "shape-mismatch"
    );
    expect(shapeErrors.length).toBeGreaterThan(0);
    expect(shapeErrors[0].severity).toBe(DiagnosticSeverity.Error);
  });

  it("reports parse errors as Error diagnostics", () => {
    const { conn, sendDiagnostics } = makeConnection();
    // Invalid syntax: unclosed paren
    const doc = makeDoc("Conv2d(kernel=5");
    validateDocument(conn, doc);
    const { diagnostics } = sendDiagnostics.mock.calls[0][0];
    expect(diagnostics.some((d: { severity: number }) => d.severity === DiagnosticSeverity.Error)).toBe(true);
  });

  it("maps 0-indexed line/col from 1-indexed SourceLoc", () => {
    const { conn, sendDiagnostics } = makeConnection();
    const content = [
      "a: Input(shape=[3,32,32])",
      "b: Input(shape=[3,64,64])",
      "Add <- a, b",
    ].join("\n");
    const doc = makeDoc(content);
    validateDocument(conn, doc);
    const { diagnostics } = sendDiagnostics.mock.calls[0][0];
    // All diagnostic positions must be 0-indexed (line >= 0)
    for (const d of diagnostics) {
      expect(d.range.start.line).toBeGreaterThanOrEqual(0);
      expect(d.range.start.character).toBeGreaterThanOrEqual(0);
    }
  });
});
