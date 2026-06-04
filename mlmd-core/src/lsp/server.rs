// ---------------------------------------------------------------------------
// LSP server — tower-lsp based MLMD Language Server
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tower_lsp::lsp_types::*;
use tower_lsp::jsonrpc::Result;
use tower_lsp::{LanguageServer, LspService, Server};

use crate::lsp::completions::get_completions;
use crate::lsp::definition::get_definition;
use crate::lsp::diagnostics::validate_document;
use crate::lsp::hover::get_hover;
use crate::lsp::references::get_references;
use crate::lsp::utf16_offset_to_byte_index;

// ---------------------------------------------------------------------------
// Document store
// ---------------------------------------------------------------------------

/// Thread-safe store of open document contents, keyed by URI.
type DocumentStore = Arc<Mutex<HashMap<Url, String>>>;

// ---------------------------------------------------------------------------
// MLMD Language Server
// ---------------------------------------------------------------------------

pub struct MlmdLanguageServer {
    client: tower_lsp::Client,
    docs: DocumentStore,
}

#[tower_lsp::async_trait]
impl LanguageServer for MlmdLanguageServer {
    // ── Initialization ────────────────────────────────────────────────
    async fn initialize(
        &self,
        _params: tower_lsp::lsp_types::InitializeParams,
    ) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::INCREMENTAL,
                )),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    ..Default::default()
                }),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "mlmd-language-server".to_string(),
                version: Some("0.1.0".to_string()),
            }),
        })
    }

    async fn initialized(&self, _params: tower_lsp::lsp_types::InitializedParams) {
        // Register all builtin blocks is expected to happen before
        // the server starts (called by the CLI entry point).
    }

    // ── Shutdown ──────────────────────────────────────────────────────
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    // ── Text Document Sync ────────────────────────────────────────────
    async fn did_open(&self, params: tower_lsp::lsp_types::DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let source = params.text_document.text;
        // Store the document content
        if let Ok(mut docs) = self.docs.lock() {
            docs.insert(uri.clone(), source.clone());
        }
        // Validate and publish diagnostics
        let diagnostics = validate_document(&source);
        self.publish_diagnostics(uri, diagnostics).await;
    }

    async fn did_change(&self, params: tower_lsp::lsp_types::DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        // Process incremental changes
        let mut source = {
            let docs = self.docs.lock().unwrap_or_else(|e| e.into_inner());
            docs.get(&uri).cloned().unwrap_or_default()
        };

        for change in &params.content_changes {
            match change.range {
                Some(range) => {
                    // Incremental update: replace the specified range
                    if let Some(new_text) = apply_incremental_change(&source, &range, &change.text) {
                        source = new_text;
                    }
                }
                None => {
                    // Full document replacement
                    source = change.text.clone();
                }
            }
        }

        // Store updated content
        if let Ok(mut docs) = self.docs.lock() {
            docs.insert(uri.clone(), source.clone());
        }

        // Validate and publish diagnostics
        let diagnostics = validate_document(&source);
        self.publish_diagnostics(uri, diagnostics).await;
    }

    async fn did_save(&self, _params: tower_lsp::lsp_types::DidSaveTextDocumentParams) {
        // No-op: validation happens on change
    }

    async fn did_close(&self, params: tower_lsp::lsp_types::DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        if let Ok(mut docs) = self.docs.lock() {
            docs.remove(&uri);
        }
        // Clear diagnostics for closed document
        self.publish_diagnostics(uri, vec![]).await;
    }

    // ── Completion ────────────────────────────────────────────────────
    async fn completion(
        &self,
        params: tower_lsp::lsp_types::CompletionParams,
    ) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let source = {
            let docs = self.docs.lock().unwrap_or_else(|e| e.into_inner());
            docs.get(&uri).cloned().unwrap_or_default()
        };

        let items = get_completions(&source, position.line, position.character);
        Ok(Some(CompletionResponse::Array(items)))
    }

    // ── Hover ─────────────────────────────────────────────────────────
    async fn hover(
        &self,
        params: tower_lsp::lsp_types::HoverParams,
    ) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let source = {
            let docs = self.docs.lock().unwrap_or_else(|e| e.into_inner());
            docs.get(&uri).cloned().unwrap_or_default()
        };

        Ok(get_hover(&source, &position))
    }

    // ── Goto Definition ───────────────────────────────────────────────
    async fn goto_definition(
        &self,
        params: tower_lsp::lsp_types::GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let source = {
            let docs = self.docs.lock().unwrap_or_else(|e| e.into_inner());
            docs.get(&uri).cloned().unwrap_or_default()
        };

        Ok(get_definition(&source, &uri, &position))
    }

    // ── References ────────────────────────────────────────────────────
    async fn references(
        &self,
        params: tower_lsp::lsp_types::ReferenceParams,
    ) -> Result<Option<Vec<Location>>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let source = {
            let docs = self.docs.lock().unwrap_or_else(|e| e.into_inner());
            docs.get(&uri).cloned().unwrap_or_default()
        };

        let locations = get_references(&source, &uri, &position);
        if locations.is_empty() {
            Ok(None)
        } else {
            Ok(Some(locations))
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

impl MlmdLanguageServer {
    async fn publish_diagnostics(&self, uri: Url, diagnostics: Vec<Diagnostic>) {
        self.client
            .send_notification::<tower_lsp::lsp_types::notification::PublishDiagnostics>(
                PublishDiagnosticsParams {
                    uri,
                    diagnostics,
                    version: None,
                },
            )
            .await;
    }
}

/// Apply an incremental text change (replace range with new text).
///
/// Both `range.start.character` and `range.end.character` are LSP UTF-16 code-unit
/// offsets; they are converted to Rust UTF-8 byte indices before slicing.
fn apply_incremental_change(source: &str, range: &Range, new_text: &str) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    let start_line = range.start.line as usize;
    let end_line = range.end.line as usize;

    if start_line >= lines.len() || end_line >= lines.len() {
        return None;
    }

    let start_byte = utf16_offset_to_byte_index(lines[start_line], range.start.character)?;
    let end_byte = utf16_offset_to_byte_index(lines[end_line], range.end.character)?;

    // Byte offsets must fall within line boundaries
    if start_byte > lines[start_line].len() || end_byte > lines[end_line].len() {
        return None;
    }

    let mut result = String::new();

    // Lines before the start line
    for line in lines.iter().take(start_line) {
        result.push_str(line);
        result.push('\n');
    }

    // Text before the change point on the start line
    result.push_str(&lines[start_line][..start_byte]);

    // Insert new text
    result.push_str(new_text);

    // Text after the change point on the end line
    result.push_str(&lines[end_line][end_byte..]);

    // Lines after the end line
    for line in lines.iter().skip(end_line + 1) {
        result.push('\n');
        result.push_str(line);
    }

    Some(result)
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

/// Start the LSP server on stdin/stdout using tower-lsp.
///
/// This function should be called from the CLI entry point.
/// Ensure `mlmd_builtin_plugins::register_all()` has been called before this.
pub async fn start_lsp_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| MlmdLanguageServer {
        client,
        docs: Arc::new(Mutex::new(HashMap::new())),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
