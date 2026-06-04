use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_char;

use crate::ast::graph::{Block, Shape};
use crate::ast::nodes::ParamValue;
use crate::block::types::ParamSpec;
use crate::plugin::traits::BlockCodegenResult;

// ---------------------------------------------------------------------------
// PluginFFI — C-ABI compatible function table for dynamic plugin loading
// ---------------------------------------------------------------------------

/// C-ABI compatible function table for a dynamically loaded plugin.
///
/// All complex data is passed as null-terminated JSON strings. The host
/// (loader) allocates input strings; the plugin allocates output strings.
/// The caller must free returned strings via the `free_string` function
/// pointer.
#[repr(C)]
pub struct PluginFFI {
    /// Get the block type name (returns a static C string).
    pub name: extern "C" fn() -> *const c_char,

    /// Get param specs as a JSON string (caller must free with `free_string`).
    pub params_json: extern "C" fn() -> *mut c_char,

    /// Infer output shapes: `inputs_json`, `params_json` → output shapes JSON.
    /// Caller frees the returned string with `free_string`.
    pub infer_shape: extern "C" fn(*const c_char, *const c_char) -> *mut c_char,

    /// Param count: `inputs_json`, `params_json` → JSON number or null.
    /// Caller frees the returned string with `free_string`.
    pub param_count: extern "C" fn(*const c_char, *const c_char) -> *mut c_char,

    /// Codegen: `target` (C string), `block_json`, `input_vars_json`,
    /// `output_vars_json` → result JSON.  Caller frees with `free_string`.
    pub codegen:
        extern "C" fn(*const c_char, *const c_char, *const c_char, *const c_char) -> *mut c_char,

    /// Free a string previously allocated by the plugin.
    pub free_string: extern "C" fn(*mut c_char),
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Allocate a Rust `CString` and return a raw pointer.
///
/// The caller must eventually call the plugin's `free_string` (or
/// [`free_rust_string`]) to deallocate the string.
pub fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

/// Free a string previously allocated with [`to_c_string`].
///
/// # Safety
///
/// `s` must be a pointer returned by [`to_c_string`] that has not already
/// been freed.
pub unsafe fn free_rust_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Parse a JSON string into a `Vec<Shape>` (for `infer_shape` inputs).
pub fn parse_shapes_json(json: &str) -> Result<Vec<Shape>, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

/// Serialize a slice of `Shape` to a JSON string (for FFI return).
pub fn shapes_to_json(shapes: &[Shape]) -> String {
    serde_json::to_string(shapes).unwrap_or_default()
}

/// Parse a JSON string into `HashMap<String, ParamValue>` (for params).
pub fn parse_params_json(json: &str) -> Result<HashMap<String, ParamValue>, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

/// Serialize a `HashMap<String, ParamValue>` to a JSON string.
pub fn params_to_json(params: &HashMap<String, ParamValue>) -> String {
    serde_json::to_string(params).unwrap_or_default()
}

/// Serialize a slice of `ParamSpec` to a JSON string.
pub fn specs_to_json(specs: &[ParamSpec]) -> String {
    serde_json::to_string(specs).unwrap_or_default()
}

/// Serialize a `BlockCodegenResult` to a JSON string.
pub fn codegen_result_to_json(result: &BlockCodegenResult) -> String {
    serde_json::to_string(result).unwrap_or_default()
}

/// Deserialize a `Block` from a JSON string.
pub fn parse_block_json(json: &str) -> Result<Block, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

/// Deserialize a `Vec<String>` from a JSON string.
pub fn parse_string_list_json(json: &str) -> Result<Vec<String>, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}
