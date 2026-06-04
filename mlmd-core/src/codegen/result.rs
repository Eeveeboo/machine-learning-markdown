pub use crate::plugin::traits::{BlockCodegenResult, CandleInit, CandleInitOrString};

// ---------------------------------------------------------------------------
// GeneratedFile
// ---------------------------------------------------------------------------

/// A single generated source file.
#[derive(Debug, Clone)]
pub struct GeneratedFile {
    /// Output path for this file (relative or absolute).
    pub path: String,
    /// Source code content.
    pub content: String,
}
