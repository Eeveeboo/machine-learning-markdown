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
 * Find all usages of a tensor name:
 * - declarations: `-> [tensorName]`
 * - join patterns: `[tensorName, ...]`
 */
export function getReferences(
  document: TextDocument,
  position: Position
): Location[] {
  const lines = document.getText().split("\n");
  const lineText = lines[position.line] ?? "";
  const tensorName = wordAtPosition(lineText, position.character);
  if (!tensorName) return [];

  const locations: Location[] = [];
  const wordRe = new RegExp(`\\b${tensorName}\\b`, "g");

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Only look inside [...] contexts (declarations and join patterns)
    const bracketRe = /\[([^\]]+)\]/g;
    let bm: RegExpExecArray | null;
    while ((bm = bracketRe.exec(line)) !== null) {
      const bracketContent = bm[1];
      const names = bracketContent.split(",").map((s) => s.trim());
      if (names.includes(tensorName)) {
        // Find the exact position of the name within this bracket group
        const bracketStart = bm.index + 1; // after `[`
        let searchFrom = bracketStart;
        let wm: RegExpExecArray | null;
        wordRe.lastIndex = 0;
        const contentWithOffset = line.slice(bracketStart);
        while ((wm = wordRe.exec(contentWithOffset)) !== null) {
          const col = bracketStart + wm.index;
          locations.push({
            uri: document.uri,
            range: Range.create(i, col, i, col + tensorName.length),
          });
        }
      }
    }
  }

  return locations;
}
