import {
  Connection,
  Diagnostic,
  DiagnosticSeverity,
} from "vscode-languageserver/node.js";
import { TextDocument } from "vscode-languageserver-textdocument";
import { tokenize, parse, buildGraph } from "../parser/index.js";
import { lint } from "../lint/index.js";
import { registry } from "../blocks/registry.js";
import type { SourceLoc } from "../ast/nodes.js";

function locToPosition(loc: SourceLoc): { line: number; character: number } {
  return {
    line: loc.line - 1,
    character: loc.col - 1,
  };
}

export function validateDocument(
  connection: Connection,
  document: TextDocument
): void {
  const text = document.getText();
  const tokens = tokenize(text);
  const { nodes, errors } = parse(tokens);
  const graph = buildGraph(nodes);
  const lintDiags = lint(graph, registry);

  const diagnostics: Diagnostic[] = [];

  // Map parse errors
  for (const err of errors) {
    const pos = locToPosition(err.loc);
    diagnostics.push({
      severity: DiagnosticSeverity.Error,
      range: {
        start: pos,
        end: { line: pos.line, character: pos.character + 1 },
      },
      message: err.message,
      source: "nnml",
    });
  }

  // Map lint diagnostics
  for (const diag of lintDiags) {
    let severity: DiagnosticSeverity;
    if (diag.severity === "error") {
      severity = DiagnosticSeverity.Error;
    } else if (diag.severity === "warning") {
      severity = DiagnosticSeverity.Warning;
    } else {
      severity = DiagnosticSeverity.Information;
    }

    const pos = locToPosition(diag.loc);
    diagnostics.push({
      severity,
      range: {
        start: pos,
        end: { line: pos.line, character: pos.character + 1 },
      },
      message: diag.message,
      source: "nnml",
      code: diag.rule,
    });
  }

  connection.sendDiagnostics({ uri: document.uri, diagnostics });
}
