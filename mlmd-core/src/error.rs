use std::fmt;

// ---------------------------------------------------------------------------
// MlmdError
// ---------------------------------------------------------------------------

/// Top-level error type for the MLMD pipeline.
#[derive(Debug, Clone)]
pub enum MlmdError {
    /// An error during parsing of MLMD source text.
    ParseError(String),
    /// An error during shape inference.
    ShapeError(String),
    /// A reference to an unknown block type.
    UnknownBlockType(String),
    /// A cycle was detected in the graph.
    CycleDetected,
    /// An error related to configuration loading or validation.
    ConfigError(String),
    /// An error originating from a plugin.
    PluginError(String),
}

impl fmt::Display for MlmdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlmdError::ParseError(msg) => write!(f, "parse error: {}", msg),
            MlmdError::ShapeError(msg) => write!(f, "shape error: {}", msg),
            MlmdError::UnknownBlockType(name) => write!(f, "unknown block type: {}", name),
            MlmdError::CycleDetected => write!(f, "cycle detected in graph"),
            MlmdError::ConfigError(msg) => write!(f, "config error: {}", msg),
            MlmdError::PluginError(msg) => write!(f, "plugin error: {}", msg),
        }
    }
}

impl std::error::Error for MlmdError {}
