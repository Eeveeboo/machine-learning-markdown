use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{LazyLock, Mutex, MutexGuard};

use crate::plugin::traits::{BlockCodegenFn, BlockPlugin};

// ---------------------------------------------------------------------------
// Global plugin registries
// ---------------------------------------------------------------------------

/// Full plugin registry: block type -> plugin instance
static PLUGIN_REGISTRY: LazyLock<Mutex<HashMap<String, Box<dyn BlockPlugin>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Per-block-per-target codegen function registry:
/// block type -> target -> codegen function
static CODGEGEN_REGISTRY: LazyLock<Mutex<HashMap<String, HashMap<String, BlockCodegenFn>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// ---------------------------------------------------------------------------
// Plugin guard for safe access
// ---------------------------------------------------------------------------

/// A guard that holds the plugin registry lock and dereferences to the
/// matched `dyn BlockPlugin`.
pub struct PluginGuard<'a> {
    guard: MutexGuard<'a, HashMap<String, Box<dyn BlockPlugin>>>,
    name: String,
}

impl<'a> Deref for PluginGuard<'a> {
    type Target = dyn BlockPlugin;
    fn deref(&self) -> &Self::Target {
        self.guard.get(&self.name).unwrap().as_ref()
    }
}

// ---------------------------------------------------------------------------
// Plugin registration and lookup
// ---------------------------------------------------------------------------

/// Register a full plugin (with shape inference, codegen, etc.)
pub fn register_plugin(name: &str, plugin: Box<dyn BlockPlugin>) {
    let mut reg = PLUGIN_REGISTRY
        .lock()
        .expect("plugin registry lock poisoned");
    reg.insert(name.to_string(), plugin);
}

/// Look up a plugin by name.
pub fn lookup_plugin(name: &str) -> Option<PluginGuard<'static>> {
    let guard = PLUGIN_REGISTRY
        .lock()
        .expect("plugin registry lock poisoned");
    if guard.contains_key(name) {
        Some(PluginGuard {
            guard,
            name: name.to_string(),
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Codegen function registration (lightweight, no full plugin needed)
// ---------------------------------------------------------------------------

/// Register a codegen function for a specific block type and target.
pub fn register_block_codegen(block_type: &str, target: &str, f: BlockCodegenFn) {
    let mut reg = CODGEGEN_REGISTRY
        .lock()
        .expect("codegen registry lock poisoned");
    reg.entry(block_type.to_string())
        .or_default()
        .insert(target.to_string(), f);
}

/// Look up a codegen function for a specific block type and target.
pub fn get_block_codegen(block_type: &str, target: &str) -> Option<BlockCodegenFn> {
    let reg = CODGEGEN_REGISTRY
        .lock()
        .expect("codegen registry lock poisoned");
    reg.get(block_type)
        .and_then(|targets| targets.get(target))
        .copied()
}
