// ---------------------------------------------------------------------------
// adapter.rs — bridges DynamicPlugin (FFI/cdylib) and WasmPlugin to the
// static registries (BlockDef + BlockCodegenFn).
//
// This allows dynamic plugin libraries (native .so/.dylib and WASM .wasm)
// to be loaded at runtime and used transparently wherever builtin blocks are
// used — shape inference, code generation, visualization, linting, etc.
//
// Usage:
//   let block_name = mlmd_core::plugin::adapter::register_dynamic_plugin(
//       "path/to/my_plugin.dylib"
//   ).expect("Failed to load plugin");
//   // The block is now registered in the global registries and can be used
//   // in .mlmd files just like any builtin block.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
#[cfg(feature = "wasm-plugins")]
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};

use crate::ast::graph::{Block, Shape};
use crate::ast::nodes::ParamValue;
use crate::block::registry::register_block;
use crate::block::types::{BlockDef, ParamSpec};
use crate::codegen::result::BlockCodegenResult;
use crate::plugin::loader::DynamicPlugin;
use crate::plugin::registry::register_block_codegen;
#[cfg(feature = "wasm-plugins")]
use crate::plugin::wasm_loader::WasmPlugin;

// ---------------------------------------------------------------------------
// Global store: block_type → DynamicPlugin
//
// The codegen registries use plain function pointers (`BlockCodegenFn`) that
// cannot capture state. We store loaded dynamic plugins here and look them
// up by block type at codegen time.
// ---------------------------------------------------------------------------

static DYNAMIC_PLUGINS: LazyLock<Mutex<HashMap<String, Arc<DynamicPlugin>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Parallel store mapping block_name → plugin library path.
/// Used by `mlmd plugin list` to display provenance.
static DYNAMIC_PLUGIN_SOURCES: LazyLock<Mutex<Vec<(String, String)>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// WASM plugin runtime source store.
#[cfg(feature = "wasm-plugins")]
static WASM_PLUGINS: LazyLock<Mutex<HashMap<String, (String, String)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// ---------------------------------------------------------------------------
// DynamicPluginBlockDef — wraps a DynamicPlugin as a BlockDef
// ---------------------------------------------------------------------------

struct DynamicPluginBlockDef {
    name: String,
    params: Vec<ParamSpec>,
    plugin: Arc<DynamicPlugin>,
}

impl BlockDef for DynamicPluginBlockDef {
    fn name(&self) -> &str {
        &self.name
    }

    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        self.plugin.infer_shape(inputs, params)
    }

    fn param_count(&self, inputs: &[Shape], params: &HashMap<String, ParamValue>) -> Option<usize> {
        self.plugin.param_count(inputs, params)
    }
}

// ---------------------------------------------------------------------------
// Static codegen dispatch functions
//
// These are registered as `BlockCodegenFn` for each target. They look up the
// appropriate `DynamicPlugin` from the global store using the block's type
// name, then delegate to the plugin's FFI codegen.
// ---------------------------------------------------------------------------

fn dynamic_pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let plugins = DYNAMIC_PLUGINS
        .lock()
        .expect("dynamic plugins lock poisoned");
    let plugin = plugins
        .get(&block.block_type)
        .expect("dynamic plugin not found for codegen");
    plugin
        .codegen("pytorch", block, input_vars, output_vars)
        .expect("dynamic plugin pytorch codegen failed")
}

fn dynamic_keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let plugins = DYNAMIC_PLUGINS
        .lock()
        .expect("dynamic plugins lock poisoned");
    let plugin = plugins
        .get(&block.block_type)
        .expect("dynamic plugin not found for codegen");
    plugin
        .codegen("keras", block, input_vars, output_vars)
        .expect("dynamic plugin keras codegen failed")
}

fn dynamic_candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let plugins = DYNAMIC_PLUGINS
        .lock()
        .expect("dynamic plugins lock poisoned");
    let plugin = plugins
        .get(&block.block_type)
        .expect("dynamic plugin not found for codegen");
    plugin
        .codegen("candle", block, input_vars, output_vars)
        .expect("dynamic plugin candle codegen failed")
}

// ---------------------------------------------------------------------------
// Public API — Native (.so/.dylib) plugins
// ---------------------------------------------------------------------------

/// Load a dynamic plugin from a `.so`/`.dylib` path and register it in the
/// global BlockDef and codegen registries.
///
/// Returns the block type name (e.g. "Scale") on success.
///
/// The library **must** export a `MLMD_PLUGIN` symbol that points to a
/// static `PluginFFI` struct.
pub fn register_dynamic_plugin(path: &str) -> Result<String, String> {
    let plugin = DynamicPlugin::load(path)?;
    let name = plugin.name().to_string();
    let params_json = plugin.params_json();

    let params: Vec<ParamSpec> = serde_json::from_str(&params_json)
        .map_err(|e| format!("failed to parse plugin params JSON for '{name}': {e}"))?;

    let plugin = Arc::new(plugin);

    // Store for codegen dispatch
    DYNAMIC_PLUGINS
        .lock()
        .expect("dynamic plugins lock poisoned")
        .insert(name.clone(), plugin.clone());

    // Record source path for `mlmd plugin list`
    DYNAMIC_PLUGIN_SOURCES
        .lock()
        .expect("dynamic plugin sources lock poisoned")
        .push((name.clone(), path.to_string()));

    // Register BlockDef (shape inference)
    register_block(Box::new(DynamicPluginBlockDef {
        name: name.clone(),
        params,
        plugin,
    }));

    // Register per-target codegen functions
    register_block_codegen(&name, "pytorch", dynamic_pytorch_codegen);
    register_block_codegen(&name, "keras", dynamic_keras_codegen);
    register_block_codegen(&name, "candle", dynamic_candle_codegen);

    Ok(name)
}

// ---------------------------------------------------------------------------
// Public API — WASM (.wasm) plugins
// ---------------------------------------------------------------------------

/// Load a WASM plugin from a `.wasm` path and register it in the global
/// BlockDef and codegen registries.
///
/// Returns the block type name on success.
///
/// The WASM module **must** export:
///   plugin_name, plugin_params_json, plugin_infer_shape,
///   plugin_param_count, plugin_codegen, plugin_free_string, plugin_alloc
#[cfg(feature = "wasm-plugins")]
pub fn register_wasm_plugin(path: &str) -> Result<String, String> {
    let mut plugin = WasmPlugin::load(Path::new(path))?;
    let name = plugin.name();
    let params_json = plugin.params_json();

    let params: Vec<ParamSpec> = serde_json::from_str(&params_json)
        .map_err(|e| format!("failed to parse plugin params JSON for '{name}': {e}"))?;

    // For WASM plugins, we need a thread-safe wrapper since WasmPlugin
    // requires &mut self for all calls.  The codegen dispatch functions
    // are static function pointers and can't capture state.
    //
    // We store the plugin path so the codegen dispatcher can re-load the
    // WASM module for each call.  This is less efficient than the native
    // case but avoids mutable global state issues.

    // Store source info
    WASM_PLUGINS
        .lock()
        .expect("wasm plugins lock poisoned")
        .insert(name.clone(), (path.to_string(), serde_json::to_string(&params).unwrap_or_default()));

    // Record in the sources list
    DYNAMIC_PLUGIN_SOURCES
        .lock()
        .expect("dynamic plugin sources lock poisoned")
        .push((name.clone(), format!("{path} (wasm)")));

    // Create a BlockDef that re-instantiates the WASM module on each call
    register_block(Box::new(WasmPluginBlockDef {
        name: name.clone(),
        params,
        path: path.to_string(),
    }));

    // Register per-target codegen functions
    register_block_codegen(&name, "pytorch", wasm_pytorch_codegen);
    register_block_codegen(&name, "keras", wasm_keras_codegen);
    register_block_codegen(&name, "candle", wasm_candle_codegen);

    Ok(name)
}

/// BlockDef adapter that re-instantiates the WASM module for each call.
#[cfg(feature = "wasm-plugins")]
struct WasmPluginBlockDef {
    name: String,
    params: Vec<ParamSpec>,
    path: String,
}

#[cfg(feature = "wasm-plugins")]
impl BlockDef for WasmPluginBlockDef {
    fn name(&self) -> &str {
        &self.name
    }

    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        let mut plugin = WasmPlugin::load(Path::new(&self.path))?;
        plugin.infer_shape(inputs, params)
    }

    fn param_count(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        let mut plugin = WasmPlugin::load(Path::new(&self.path)).ok()?;
        plugin.param_count(inputs, params)
    }
}

/// Helper: reload a WASM plugin and call its codegen.
#[cfg(feature = "wasm-plugins")]
fn wasm_codegen(
    target: &str,
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let wasm_plugins = WASM_PLUGINS
        .lock()
        .expect("wasm plugins lock poisoned");
    let (path, _) = wasm_plugins
        .get(&block.block_type)
        .expect("WASM plugin not found for codegen");
    let mut plugin = WasmPlugin::load(Path::new(path))
        .expect("failed to reload WASM plugin for codegen");
    plugin
        .codegen(target, block, input_vars, output_vars)
        .expect("WASM plugin codegen failed")
}

#[cfg(feature = "wasm-plugins")]
fn wasm_pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    wasm_codegen("pytorch", block, input_vars, output_vars)
}

#[cfg(feature = "wasm-plugins")]
fn wasm_keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    wasm_codegen("keras", block, input_vars, output_vars)
}

#[cfg(feature = "wasm-plugins")]
fn wasm_candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    wasm_codegen("candle", block, input_vars, output_vars)
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Detect whether a plugin path points to a WASM module (`.wasm` extension).
pub fn is_wasm_path(path: &str) -> bool {
    path.ends_with(".wasm")
}

/// Register a plugin from a path, auto-detecting native vs WASM format.
///
/// Returns the block type name on success.
pub fn register_plugin_from_path(path: &str) -> Result<String, String> {
    if is_wasm_path(path) {
        #[cfg(feature = "wasm-plugins")]
        {
            register_wasm_plugin(path)
        }
        #[cfg(not(feature = "wasm-plugins"))]
        {
            Err(format!(
                "WASM plugins are not supported in this build. \
                 Rebuild with the 'wasm-plugins' feature enabled. \
                 Path: {path}"
            ))
        }
    } else {
        register_dynamic_plugin(path)
    }
}

/// Return all registered dynamic plugins as `(block_name, library_path)` pairs.
pub fn all_dynamic_plugins() -> Vec<(String, String)> {
    DYNAMIC_PLUGIN_SOURCES
        .lock()
        .expect("dynamic plugin sources lock poisoned")
        .clone()
}
