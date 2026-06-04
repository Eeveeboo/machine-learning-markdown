use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;
use std::sync::Arc;

use libloading::{Library, Symbol};

use crate::ast::graph::{Block, Shape};
use crate::ast::nodes::ParamValue;
use crate::codegen::result::BlockCodegenResult;
use crate::plugin::ffi::PluginFFI;

// ---------------------------------------------------------------------------
// DynamicPlugin — wraps a .so / .dylib loaded at runtime
// ---------------------------------------------------------------------------

/// A dynamically loaded plugin instance backed by a shared library.
pub struct DynamicPlugin {
    /// Keep the library alive for the plugin's lifetime.
    _lib: Arc<Library>,
    /// Reference to the C-ABI function table exported by the library.
    ffi: &'static PluginFFI,
}

impl DynamicPlugin {
    /// Load a plugin from a `.so` / `.dylib` file.
    ///
    /// The library **must** export a `MLMD_PLUGIN` function with the signature
    /// `extern "C" fn() -> *const PluginFFI`.  The loader calls this function
    /// to obtain the FFI function table.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        unsafe {
            let lib = Arc::new(Library::new(path.as_ref()).map_err(|e| {
                format!(
                    "Failed to load library '{}': {}",
                    path.as_ref().display(),
                    e
                )
            })?);

            // MLMD_PLUGIN is an extern "C" fn() -> *const PluginFFI.
            // Load it as a function pointer, then call it to get the table.
            type PluginInitFn = unsafe extern "C" fn() -> *const PluginFFI;

            let init_fn: Symbol<PluginInitFn> = lib
                .get(b"MLMD_PLUGIN\0")
                .map_err(|e| format!("Failed to locate MLMD_PLUGIN symbol: {}", e))?;

            let ffi_ptr: *const PluginFFI = init_fn();
            if ffi_ptr.is_null() {
                return Err("MLMD_PLUGIN returned null pointer".to_string());
            }

            let ffi_ref: &'static PluginFFI = &*ffi_ptr;

            Ok(DynamicPlugin {
                _lib: lib,
                ffi: ffi_ref,
            })
        }
    }

    /// Return the block type name advertised by this plugin.
    pub fn name(&self) -> &'static str {
        unsafe {
            let ptr = (self.ffi.name)();
            if ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
            }
        }
    }

    /// Get parameter specifications as a JSON string.
    ///
    /// The returned value is a `Vec` of [`ParamSpec`](crate::block::types::ParamSpec) serialized to
    /// JSON.  The caller should parse it with `serde_json`.
    pub fn params_json(&self) -> String {
        unsafe {
            let ptr = (self.ffi.params_json)();
            if ptr.is_null() {
                return String::new();
            }
            let s = CStr::from_ptr(ptr).to_str().unwrap_or("").to_string();
            (self.ffi.free_string)(ptr);
            s
        }
    }

    /// Infer output shapes given input shapes and resolved parameters.
    pub fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        let inputs_json =
            serde_json::to_string(inputs).map_err(|e| format!("serialize inputs: {}", e))?;
        let params_json =
            serde_json::to_string(params).map_err(|e| format!("serialize params: {}", e))?;

        let inputs_c = crate::plugin::ffi::to_c_string(&inputs_json);
        let params_c = crate::plugin::ffi::to_c_string(&params_json);

        unsafe {
            let result_ptr = (self.ffi.infer_shape)(inputs_c, params_c);

            // Free the C strings we allocated
            crate::plugin::ffi::free_rust_string(inputs_c);
            crate::plugin::ffi::free_rust_string(params_c);

            if result_ptr.is_null() {
                return Err("Null result from infer_shape".to_string());
            }

            let result_str = CStr::from_ptr(result_ptr)
                .to_str()
                .map_err(|e| format!("invalid UTF-8 in infer_shape result: {}", e))?
                .to_string();

            // Free the plugin-allocated result string
            (self.ffi.free_string)(result_ptr);

            serde_json::from_str(&result_str)
                .map_err(|e| format!("deserialize infer_shape result: {}", e))
        }
    }

    /// Compute total parameter count given input shapes and params.
    /// Returns `None` if the plugin returned a null/empty JSON value.
    pub fn param_count(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        let inputs_json = serde_json::to_string(inputs).unwrap_or_default();
        let params_json = serde_json::to_string(params).unwrap_or_default();

        let inputs_c = crate::plugin::ffi::to_c_string(&inputs_json);
        let params_c = crate::plugin::ffi::to_c_string(&params_json);

        unsafe {
            let result_ptr = (self.ffi.param_count)(inputs_c, params_c);

            crate::plugin::ffi::free_rust_string(inputs_c);
            crate::plugin::ffi::free_rust_string(params_c);

            if result_ptr.is_null() {
                return None;
            }

            let result_str = CStr::from_ptr(result_ptr)
                .to_str()
                .unwrap_or("null")
                .to_string();

            (self.ffi.free_string)(result_ptr);

            // Accept both a JSON number and a string-encoded number
            if result_str.is_empty() || result_str == "null" {
                None
            } else {
                // Try parsing as JSON number first, then as plain string
                serde_json::from_str::<serde_json::Value>(&result_str)
                    .ok()
                    .and_then(|v| v.as_u64().map(|n| n as usize))
                    .or_else(|| result_str.parse::<usize>().ok())
            }
        }
    }

    /// Generate code for this block for a specific target.
    pub fn codegen(
        &self,
        target: &str,
        block: &Block,
        input_vars: &[String],
        output_vars: &[String],
    ) -> Option<BlockCodegenResult> {
        let target_c = crate::plugin::ffi::to_c_string(target);
        let block_json = serde_json::to_string(block).unwrap_or_default();
        let block_c = crate::plugin::ffi::to_c_string(&block_json);
        let input_vars_json = serde_json::to_string(input_vars).unwrap_or_default();
        let input_vars_c = crate::plugin::ffi::to_c_string(&input_vars_json);
        let output_vars_json = serde_json::to_string(output_vars).unwrap_or_default();
        let output_vars_c = crate::plugin::ffi::to_c_string(&output_vars_json);

        unsafe {
            let result_ptr = (self.ffi.codegen)(target_c, block_c, input_vars_c, output_vars_c);

            crate::plugin::ffi::free_rust_string(target_c);
            crate::plugin::ffi::free_rust_string(block_c);
            crate::plugin::ffi::free_rust_string(input_vars_c);
            crate::plugin::ffi::free_rust_string(output_vars_c);

            if result_ptr.is_null() {
                return None;
            }

            let result_str = CStr::from_ptr(result_ptr)
                .to_str()
                .unwrap_or("null")
                .to_string();

            (self.ffi.free_string)(result_ptr);

            serde_json::from_str(&result_str).ok()
        }
    }
}

// ---------------------------------------------------------------------------
// PluginHandle — unified way to refer to built-in or dynamic plugins
// ---------------------------------------------------------------------------

/// A handle that can refer to either a built-in block type name or a
/// dynamically loaded plugin.
pub enum PluginHandle {
    /// Builtin block type name (looked up in the static registry).
    Builtin(&'static str),
    /// Dynamically loaded plugin instance.
    Dynamic(DynamicPlugin),
}

impl PluginHandle {
    /// Return the block type name.
    pub fn name(&self) -> &str {
        match self {
            PluginHandle::Builtin(name) => name,
            PluginHandle::Dynamic(p) => p.name(),
        }
    }
}
