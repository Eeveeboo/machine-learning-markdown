use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use crate::ast::graph::Graph;
use crate::codegen::result::GeneratedFile;

// ---------------------------------------------------------------------------
// CodegenTarget trait
// ---------------------------------------------------------------------------

/// A codegen target produces source files from a graph IR.
pub trait CodegenTarget: Send + Sync {
    /// Human-readable name, e.g. "pytorch", "candle".
    fn name(&self) -> &'static str;

    /// File extension for generated source files, e.g. "py", "rs".
    fn file_extension(&self) -> &'static str;

    /// Generate source files for the given graph.
    fn generate(&self, graph: &Graph) -> Vec<GeneratedFile>;
}

// ---------------------------------------------------------------------------
// Global target registry
// ---------------------------------------------------------------------------

static TARGET_REGISTRY: LazyLock<Mutex<HashMap<String, Arc<dyn CodegenTarget>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Register a codegen target.
pub fn register_target(target: Box<dyn CodegenTarget>) {
    let name = target.name().to_string();
    let mut reg = TARGET_REGISTRY
        .lock()
        .expect("target registry lock poisoned");
    reg.insert(name, Arc::from(target));
}

/// Look up a codegen target by name. Returns an `Arc` to the registered target.
pub fn get_target(name: &str) -> Option<Arc<dyn CodegenTarget>> {
    let reg = TARGET_REGISTRY
        .lock()
        .expect("target registry lock poisoned");
    reg.get(name).map(|t| t.clone())
}

/// Invoke a closure with a reference to the registered target.
/// Returns `None` if no target with that name exists.
pub fn with_target<F, R>(name: &str, f: F) -> Option<R>
where
    F: FnOnce(&dyn CodegenTarget) -> R,
{
    let reg = TARGET_REGISTRY
        .lock()
        .expect("target registry lock poisoned");
    reg.get(name).map(|t| f(t.as_ref()))
}
