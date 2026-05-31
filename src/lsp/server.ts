import {
  createConnection,
  TextDocuments,
  InitializeResult,
  TextDocumentSyncKind,
} from "vscode-languageserver/node.js";
import { TextDocument } from "vscode-languageserver-textdocument";
import { validateDocument } from "./diagnostics.js";
import { getCompletions } from "./completions.js";
import { getHover } from "./hover.js";
import { getDefinition } from "./definition.js";
import { getReferences } from "./references.js";

export function startServer(): void {
  const connection = createConnection();
  const documents = new TextDocuments(TextDocument);

  connection.onInitialize((): InitializeResult => {
    return {
      capabilities: {
        textDocumentSync: TextDocumentSyncKind.Incremental,
        completionProvider: {
          resolveProvider: false,
        },
        hoverProvider: true,
        definitionProvider: true,
        referencesProvider: true,
      },
    };
  });

  documents.onDidChangeContent((change) => {
    validateDocument(connection, change.document);
  });

  connection.onCompletion((params) => {
    const doc = documents.get(params.textDocument.uri);
    if (!doc) return [];
    return getCompletions(doc, params.position);
  });

  connection.onHover((params) => {
    const doc = documents.get(params.textDocument.uri);
    if (!doc) return null;
    return getHover(doc, params.position);
  });

  connection.onDefinition((params) => {
    const doc = documents.get(params.textDocument.uri);
    if (!doc) return null;
    return getDefinition(doc, params.position);
  });

  connection.onReferences((params) => {
    const doc = documents.get(params.textDocument.uri);
    if (!doc) return [];
    return getReferences(doc, params.position);
  });

  documents.listen(connection);
  connection.listen();
}
