// ---------------------------------------------------------------------------
// LSP goto-definition — jump from tensor reference to its declaration
// ---------------------------------------------------------------------------

use tower_lsp::lsp_types::{GotoDefinitionResponse, Location, Position, Range, Url};

use crate::lsp::word_at_position;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find the declaration location of the tensor name under the cursor.
///
/// Searches the document for `-> [tensorName]` or `-> [..., tensorName, ...]`
/// and returns the position where the tensor is declared.
pub fn get_definition(
    doc_source: &str,
    uri: &Url,
    position: &Position,
) -> Option<GotoDefinitionResponse> {
    let lines: Vec<&str> = doc_source.lines().collect();
    let line = lines.get(position.line as usize)?;
    let tensor_name = word_at_position(line, position.character)?;
    if tensor_name.is_empty() {
        return None;
    }

    // Search all lines for `-> [...]` declarations
    for (line_idx, line_text) in lines.iter().enumerate() {
        // Find all `-> [...]` patterns on this line
        let mut search_pos = 0;
        while let Some(arrow_pos) = line_text[search_pos..].find("->") {
            let abs_arrow = search_pos + arrow_pos;
            let after_arrow = abs_arrow + 2;

            // Find opening bracket after ->
            if let Some(bracket_pos) = line_text[after_arrow..].find('[') {
                let abs_bracket = after_arrow + bracket_pos;
                let bracket_content_start = abs_bracket + 1;

                // Find closing bracket
                if let Some(close_pos) = line_text[bracket_content_start..].find(']') {
                    let abs_close = bracket_content_start + close_pos;
                    let content = &line_text[bracket_content_start..abs_close];

                    // Check if our tensor name is in this bracket list
                    for part in content.split(',') {
                        let trimmed = part.trim();
                        if trimmed == tensor_name {
                            // Find the exact column of the tensor name within the content
                            let col = if let Some(name_pos) = content.find(&tensor_name) {
                                bracket_content_start + name_pos
                            } else {
                                bracket_content_start
                            };
                            return Some(GotoDefinitionResponse::Scalar(Location {
                                uri: uri.clone(),
                                range: Range {
                                    start: Position {
                                        line: line_idx as u32,
                                        character: col as u32,
                                    },
                                    end: Position {
                                        line: line_idx as u32,
                                        character: (col + tensor_name.len()) as u32,
                                    },
                                },
                            }));
                        }
                    }
                    search_pos = abs_close + 1;
                } else {
                    search_pos = abs_bracket + 1;
                }
            } else {
                search_pos = after_arrow;
            }
        }
    }

    None
}
