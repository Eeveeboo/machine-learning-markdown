use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use crate::block::types::BlockDef;
use crate::shape::BlockRegistry;

// ---------------------------------------------------------------------------
// Global block registry
// ---------------------------------------------------------------------------

static REGISTRY: LazyLock<Mutex<HashMap<String, Arc<dyn BlockDef>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A guard that holds the registry lock and dereferences to the matched
/// `dyn BlockDef`.
pub struct BlockDefGuard<'a> {
    guard: MutexGuard<'a, HashMap<String, Arc<dyn BlockDef>>>,
    name: String,
}

impl<'a> Deref for BlockDefGuard<'a> {
    type Target = dyn BlockDef;
    fn deref(&self) -> &Self::Target {
        // SAFETY: we only construct this guard when the key exists
        self.guard.get(&self.name).unwrap().as_ref()
    }
}

/// Register a block definition under its name.
pub fn register_block(def: Box<dyn BlockDef>) {
    let name = def.name().to_string();
    let mut registry = REGISTRY.lock().expect("block registry lock poisoned");
    registry.insert(name, Arc::from(def));
}

/// Look up a block definition by name.
/// Returns `None` if no block with that name is registered.
pub fn lookup_block(name: &str) -> Option<BlockDefGuard<'static>> {
    let guard = REGISTRY.lock().expect("block registry lock poisoned");
    if guard.contains_key(name) {
        Some(BlockDefGuard {
            guard,
            name: name.to_string(),
        })
    } else {
        None
    }
}

/// Build a `BlockRegistry` from the global registry (clones the `Arc`s).
/// This is useful for passing to `lint()` or `infer_shapes()`.
pub fn build_block_registry() -> BlockRegistry {
    let guard = REGISTRY.lock().expect("block registry lock poisoned");
    let mut registry = BlockRegistry::new();
    for (_name, def) in guard.iter() {
        registry.register_arc(def.clone());
    }
    registry
}

/// Return all registered block type names.
pub fn all_block_names() -> Vec<String> {
    let guard = REGISTRY.lock().expect("block registry lock poisoned");
    guard.keys().cloned().collect()
}
