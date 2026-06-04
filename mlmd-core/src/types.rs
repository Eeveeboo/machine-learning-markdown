// ---------------------------------------------------------------------------
// Re-exports of public types for convenient use by downstream crates.
// ---------------------------------------------------------------------------

pub use crate::ast::graph::{Block, Edge, Graph, Group, Shape};
pub use crate::ast::nodes::*;
pub use crate::block::registry::{lookup_block, register_block};
pub use crate::block::types::{BlockDef, ParamSpec, ParamType};
pub use crate::codegen::result::GeneratedFile;
pub use crate::codegen::target::{register_target, CodegenTarget};
pub use crate::config::loader::MlmdConfig;
pub use crate::error::MlmdError;
pub use crate::plugin::registry::{get_block_codegen, register_block_codegen};
pub use crate::plugin::traits::{
    BlockCodegenFn, BlockCodegenResult, BlockPlugin, CandleInit, CandleInitOrString,
};
