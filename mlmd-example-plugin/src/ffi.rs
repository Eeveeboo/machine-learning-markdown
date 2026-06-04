// ---------------------------------------------------------------------------
// mlmd-example-plugin/src/ffi.rs – C-compatible FFI entry point.
//
// This module exposes a `MLMD_PLUGIN` symbol that a `DynamicPlugin` loader
// (see `mlmd_core::plugin::loader`) can locate via `libloading`.
//
// Unlike the builtin-plugins crate (which bundles 43 block types and returns
// null for per-block operations), this plugin exports a single block type
// ("Scale") and provides working implementations for all FFI functions.
//
// This is the **canonical reference** for dynamic plugin authors.
// ---------------------------------------------------------------------------

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::OnceLock;

use mlmd_core::plugin::ffi::{free_rust_string, parse_shapes_json,
                             shapes_to_json, to_c_string, PluginFFI};
use mlmd_core::plugin::traits::BlockCodegenResult;

// ---------------------------------------------------------------------------
// Global FFI table – initialised exactly once
// ---------------------------------------------------------------------------

static PLUGIN_FFI: OnceLock<PluginFFI> = OnceLock::new();

fn get_ffi() -> &'static PluginFFI {
    PLUGIN_FFI.get_or_init(|| {
        // Register the Scale block + codegen with the global registries
        crate::register();

        PluginFFI {
            name: name_fn,
            params_json: params_json_fn,
            infer_shape: infer_shape_fn,
            param_count: param_count_fn,
            codegen: codegen_fn,
            free_string: free_string_fn,
        }
    })
}

// ---------------------------------------------------------------------------
// Exported symbol – discovered by DynamicPlugin::load
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn MLMD_PLUGIN() -> *const PluginFFI {
    get_ffi() as *const PluginFFI
}

// ---------------------------------------------------------------------------
// PluginFFI function implementations
// ---------------------------------------------------------------------------

/// Return the block type name as a static C string.
extern "C" fn name_fn() -> *const c_char {
    c"Scale".as_ptr()
}

/// Return parameter specifications as a JSON string.
///
/// # Allocation
/// The returned string is heap-allocated. The caller must free it
/// via the `free_string` function pointer.
extern "C" fn params_json_fn() -> *mut c_char {
    // Matches the ParamSpec from lib.rs: [{"name":"factor","param_type":"Number","required":true,"default":null}]
    to_c_string(r#"[{"name":"factor","param_type":"Number","required":true,"default":null}]"#)
}

/// Infer shapes given serialized inputs and params.
///
/// # Safety
/// `inputs_json` and `params_json` must be valid null-terminated UTF-8 strings.
extern "C" fn infer_shape_fn(
    inputs_json: *const c_char,
    params_json: *const c_char,
) -> *mut c_char {
    let inputs_str = unsafe {
        if inputs_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(inputs_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let _params_str = unsafe {
        if params_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(params_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let inputs = match parse_shapes_json(inputs_str) {
        Ok(v) => v,
        Err(_) => return std::ptr::null_mut(),
    };

    if inputs.is_empty() {
        return std::ptr::null_mut();
    }

    // Scale: passthrough shape
    let output_shapes = vec![inputs[0].clone()];
    to_c_string(&shapes_to_json(&output_shapes))
}

/// Compute parameter count.
///
/// # Safety
/// Same contract as `infer_shape_fn`.
extern "C" fn param_count_fn(
    inputs_json: *const c_char,
    params_json: *const c_char,
) -> *mut c_char {
    let _inputs_str = unsafe {
        if inputs_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(inputs_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let _params_str = unsafe {
        if params_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(params_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    // Scale always has 1 learnable parameter
    to_c_string("1")
}

/// Generate code for the specified target.
///
/// # Safety
/// All four pointer arguments must be valid null-terminated UTF-8 strings.
extern "C" fn codegen_fn(
    target: *const c_char,
    block_json: *const c_char,
    input_vars_json: *const c_char,
    output_vars_json: *const c_char,
) -> *mut c_char {
    let target_str = unsafe {
        if target.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(target).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let block_str = unsafe {
        if block_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(block_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let input_vars_str = unsafe {
        if input_vars_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(input_vars_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let output_vars_str = unsafe {
        if output_vars_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(output_vars_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let block: mlmd_core::ast::graph::Block = match serde_json::from_str(block_str) {
        Ok(b) => b,
        Err(_) => return std::ptr::null_mut(),
    };

    let input_vars: Vec<String> = match serde_json::from_str(input_vars_str) {
        Ok(v) => v,
        Err(_) => return std::ptr::null_mut(),
    };

    let output_vars: Vec<String> = match serde_json::from_str(output_vars_str) {
        Ok(v) => v,
        Err(_) => return std::ptr::null_mut(),
    };

    let result: BlockCodegenResult = match target_str {
        "pytorch" => super::pytorch_codegen(&block, &input_vars, &output_vars),
        "keras" => super::keras_codegen(&block, &input_vars, &output_vars),
        "candle" => super::candle_codegen(&block, &input_vars, &output_vars),
        _ => return std::ptr::null_mut(),
    };

    match serde_json::to_string(&result) {
        Ok(json) => to_c_string(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a string previously allocated by one of the plugin functions.
extern "C" fn free_string_fn(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            free_rust_string(s);
        }
    }
}
