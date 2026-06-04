// ---------------------------------------------------------------------------
// adapter.rs — bridges DynamicPlugin (FFI/cdylib) to the static registries
// (BlockDef + BlockCodegenFn).
//
// This allows dynamic plugin libraries to be loaded at runtime and used
// transparently wherever builtin blocks are used — shape inference, code
// generation, visualization, linting, etc.
//
// Usage:
//   let block_name = mlmd_core::plugin::adapter::register_dynamic_plugin(
//       "path/to/my_plugin.dylib"
//   ).expect("Failed to load plugin");
//   // The block is now registered in the global registries and can be used
//   // in .mlmd files just like any builtin block.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use crate::ast::graph::{Block, Shape};
use crate::ast::nodes::ParamValue;
use crate::block::registry::register_block;
use crate::block::types::{BlockDef, ParamSpec};
use crate::codegen::result::BlockCodegenResult;
use crate::plugin::loader::DynamicPlugin;
use crate::plugin::registry::register_block_codegen;

// ---------------------------------------------------------------------------
// Global store: block_type → DynamicPlugin
//
// The codegen registries use plain function pointers (`BlockCodegenFn`) that
// cannot capture state. We store loaded dynamic plugins here and look them
// up by block type at codegen time.
// ---------------------------------------------------------------------------

static DYNAMIC_PLUGINS: LazyLock<Mutex<HashMap<String, Arc<DynamicPlugin>>>> =
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

    fn param_count(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
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
// Public API
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

    let params: Vec<ParamSpec> =
        serde_json::from_str(&params_json).map_err(|e| {
            format!("failed to parse plugin params JSON for '{name}': {e}")
        })?;

    let plugin = Arc::new(plugin);

    // Store for codegen dispatch
    DYNAMIC_PLUGINS
        .lock()
        .expect("dynamic plugins lock poisoned")
        .insert(name.clone(), plugin.clone());

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
