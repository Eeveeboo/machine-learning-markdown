// ---------------------------------------------------------------------------
// LSP diagnostics — validate MLMD documents and produce LSP diagnostics
// ---------------------------------------------------------------------------

use tower_lsp::lsp_types::{
    Diagnostic, DiagnosticSeverity, Position, Range,
};

use crate::ast::nodes::SourceLoc;
use crate::block::registry::build_block_registry;
use crate::lint::Severity;
use crate::parser::graph_builder::build_graph;
use crate::parser::parser::parse;
use crate::parser::tokenizer::tokenize;

// ---------------------------------------------------------------------------
// SourceLoc → LSP Position conversion
// ---------------------------------------------------------------------------

/// Convert a `SourceLoc` (1-based) to an LSP `Position` (0-based).
pub fn source_loc_to_position(loc: &SourceLoc) -> Position {
    Position {
        line: loc.line.saturating_sub(1) as u32,
        character: loc.col.saturating_sub(1) as u32,
    }
}

// ---------------------------------------------------------------------------
// Diagnostics conversion
// ---------------------------------------------------------------------------

/// Convert a `LintDiagnostic` from our lint module to an LSP `Diagnostic`.
pub fn to_lsp_diagnostic(
    d: &crate::lint::LintDiagnostic,
) -> Diagnostic {
    Diagnostic {
        range: Range {
            start: source_loc_to_position(&d.loc),
            end: source_loc_to_position(&d.loc),
        },
        severity: Some(match d.severity {
            Severity::Error => DiagnosticSeverity::ERROR,
            Severity::Warning => DiagnosticSeverity::WARNING,
        }),
        message: d.message.clone(),
        source: Some("mlmd".to_string()),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Document validation
// ---------------------------------------------------------------------------

/// Validate an MLMD source string and return a list of LSP diagnostics.
///
/// Steps:
/// 1. Tokenize
/// 2. Parse (collects parse errors)
/// 3. If parsing succeeded, build graph and run lint
/// 4. Convert all diagnostics to LSP format
pub fn validate_document(source: &str) -> Vec<Diagnostic> {
    let tokens = tokenize(source);
    let parse_result = parse(&tokens);

    let mut diagnostics: Vec<Diagnostic> = Vec::new();

    // Parse errors become diagnostics
    for err in &parse_result.errors {
        let pos = source_loc_to_position(&err.loc);
        diagnostics.push(Diagnostic {
            range: Range {
                start: pos,
                end: Position {
                    line: pos.line,
                    character: pos.character + 1,
                },
            },
            severity: Some(DiagnosticSeverity::ERROR),
            message: err.message.clone(),
            source: Some("mlmd".to_string()),
            ..Default::default()
        });
    }

    // Only lint if parsing succeeded
    if parse_result.errors.is_empty() {
        let graph = build_graph(&parse_result.nodes);
        let registry = build_block_registry();
        let lint_diags = crate::lint::lint(&graph, &registry);
        for d in &lint_diags {
            diagnostics.push(to_lsp_diagnostic(d));
        }
    }

    diagnostics
}
