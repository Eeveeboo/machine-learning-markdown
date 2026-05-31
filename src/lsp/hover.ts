import { Hover, Position, MarkupKind } from "vscode-languageserver/node.js";
import { TextDocument } from "vscode-languageserver-textdocument";
import { registry } from "../blocks/registry.js";

function wordAtPosition(lineText: string, char: number): string {
  let start = char;
  let end = char;
  while (start > 0 && /\w/.test(lineText[start - 1])) start--;
  while (end < lineText.length && /\w/.test(lineText[end])) end++;
  return lineText.slice(start, end);
}

export function getHover(
  document: TextDocument,
  position: Position,
  reg: Map<string, { name: string; params: { name: string; type: string; required: boolean }[] }> = registry
): Hover | null {
  const lines = document.getText().split("\n");
  const lineText = lines[position.line] ?? "";
  const word = wordAtPosition(lineText, position.character);
  if (!word) return null;

  const def = reg.get(word);
  if (!def) return null;

  const paramList = def.params
    .map((p) => `  ${p.name}: ${p.type}${p.required ? "" : "?"}`)
    .join("\n");

  const contents = `**${def.name}**\n\`\`\`\nparams:\n${paramList || "  (none)"}\n\`\`\``;

  return {
    contents: {
      kind: MarkupKind.Markdown,
      value: contents,
    },
  };
}
