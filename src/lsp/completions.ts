import { CompletionItem, CompletionItemKind, Position } from "vscode-languageserver/node.js";
import { TextDocument } from "vscode-languageserver-textdocument";
import { registry } from "../blocks/registry.js";

/**
 * Extract all tensor names declared via `-> [tensorName]` in the document.
 */
function getNamedTensors(document: TextDocument): string[] {
  const text = document.getText();
  const names: string[] = [];
  const re = /->\s*\[([^\]]+)\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    for (const part of m[1].split(",")) {
      const name = part.trim();
      if (name) names.push(name);
    }
  }
  return names;
}

/**
 * Return the block type name on the current line (before `(`), if any.
 */
function getBlockTypeOnLine(lineText: string, charOffset: number): string | null {
  // Look for something like `BlockType(` before the cursor
  const before = lineText.slice(0, charOffset);
  const m = before.match(/(\w+)\s*\([^)]*$/);
  return m ? m[1] : null;
}

export function getCompletions(
  document: TextDocument,
  position: Position,
  reg: Map<string, { name: string; params: { name: string }[] }> = registry
): CompletionItem[] {
  const lines = document.getText().split("\n");
  const lineText = lines[position.line] ?? "";
  const charOffset = position.character;
  const before = lineText.slice(0, charOffset);

  // Inside `[...]` context → suggest tensor names
  const openBracket = before.lastIndexOf("[");
  const closeBracket = before.lastIndexOf("]");
  if (openBracket !== -1 && openBracket > closeBracket) {
    const tensors = getNamedTensors(document);
    return tensors.map((name) => ({
      label: name,
      kind: CompletionItemKind.Variable,
    }));
  }

  // Inside `(...)` context → suggest param names for that block type
  const openParen = before.lastIndexOf("(");
  const closeParen = before.lastIndexOf(")");
  if (openParen !== -1 && openParen > closeParen) {
    const blockType = getBlockTypeOnLine(lineText, charOffset);
    if (blockType) {
      const def = reg.get(blockType);
      if (def) {
        return def.params.map((p) => ({
          label: p.name,
          kind: CompletionItemKind.Property,
          insertText: `${p.name}=`,
        }));
      }
    }
  }

  // At start of line or after `->`: suggest block type names
  const trimmed = before.trimStart();
  const afterArrow = /->[ \t]*[\w]*$/.test(before);
  const atLineStart = /^[\w]*$/.test(trimmed);
  if (atLineStart || afterArrow) {
    return Array.from(reg.values()).map((def) => ({
      label: def.name,
      kind: CompletionItemKind.Class,
    }));
  }

  return [];
}
