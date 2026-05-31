import { Location, Position, Range } from "vscode-languageserver/node.js";
import { TextDocument } from "vscode-languageserver-textdocument";

function wordAtPosition(lineText: string, char: number): string {
  let start = char;
  let end = char;
  while (start > 0 && /\w/.test(lineText[start - 1])) start--;
  while (end < lineText.length && /\w/.test(lineText[end])) end++;
  return lineText.slice(start, end);
}

/**
 * Find the line where `-> [tensorName]` declares the given tensor.
 */
export function getDefinition(
  document: TextDocument,
  position: Position
): Location | null {
  const lines = document.getText().split("\n");
  const lineText = lines[position.line] ?? "";
  const tensorName = wordAtPosition(lineText, position.character);
  if (!tensorName) return null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Match -> [name] or -> [name, ...]
    const re = /->\s*\[([^\]]+)\]/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(line)) !== null) {
      const names = m[1].split(",").map((s) => s.trim());
      if (names.includes(tensorName)) {
        const col = line.indexOf(tensorName, m.index);
        return {
          uri: document.uri,
          range: Range.create(i, col, i, col + tensorName.length),
        };
      }
    }
  }

  return null;
}
