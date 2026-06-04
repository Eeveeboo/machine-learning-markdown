// ---------------------------------------------------------------------------
// mlmd-builtin-plugins/src/ffi.rs – C-compatible FFI entry point.
//
// This module exposes a `MLMD_PLUGIN` symbol that a `DynamicPlugin` loader
// (see `mlmd_core::plugin::loader`) can locate via `libloading`.
//
// The primary purpose of the cdylib is to:
//   1. Register all 43 builtin block types with the global block registry
//      when the library is first loaded.
//   2. Provide a `PluginFFI` table that external consumers can use to query
//      the plugin name and parameter specifications.
//
// Because this single plugin contains 43 different block types, the
// `infer_shape`, `param_count`, and `codegen` functions — which require a
// block-type discriminator — return null (indicating "not applicable").
// External consumers that load this cdylib should use the static registry
// (`lookup_block`) to dispatch by block type after loading.
// ---------------------------------------------------------------------------

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::OnceLock;

use mlmd_core::plugin::ffi::{free_rust_string, to_c_string, PluginFFI};

// ---------------------------------------------------------------------------
// Global FFI table – initialised exactly once
// ---------------------------------------------------------------------------

static PLUGIN_FFI: OnceLock<PluginFFI> = OnceLock::new();

/// Initialise the FFI table with the actual function pointers.
/// Called automatically when the library is loaded via `MLMD_PLUGIN`.
fn get_ffi() -> &'static PluginFFI {
    PLUGIN_FFI.get_or_init(|| {
        // Register all builtin block types with the global registry.
        crate::register_all();

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

/// Entry point looked up by `DynamicPlugin::load` via `libloading`.
///
/// Returns a pointer to a static `PluginFFI` table.  The caller must not
/// free or otherwise modify the pointed-to memory.
///
/// # Safety
///
/// The returned pointer is valid for the lifetime of the loaded library.
/// No other safety concerns — this simply returns a reference to a global
/// static that will never change.
#[no_mangle]
pub extern "C" fn MLMD_PLUGIN() -> *const PluginFFI {
    get_ffi() as *const PluginFFI
}

// ---------------------------------------------------------------------------
// PluginFFI function implementations
// ---------------------------------------------------------------------------

/// Return the plugin name as a static C string.
///
/// No heap allocation is performed; the returned pointer points into
/// the binary's `.rodata` section and must NOT be freed.
extern "C" fn name_fn() -> *const c_char {
    c"mlmd_builtin_plugins".as_ptr()
}

/// Return an empty JSON array as the parameter specification.
///
/// Builtin blocks register their own `ParamSpec` entries through the
/// static registry, so there are no plugin-level parameters.
///
/// # Allocation
///
/// The returned string is heap-allocated.  The caller must free it
/// via the `free_string` function pointer.
extern "C" fn params_json_fn() -> *mut c_char {
    to_c_string("[]")
}

/// Infer output shapes for a block type.
///
/// **Note**: This plugin contains 43 block types, but the standard
/// `PluginFFI` interface does not pass a block-type discriminator.
/// External consumers should call `lookup_block()` after loading.
///
/// Returns `null` to indicate the operation is not applicable through
/// this interface.
///
/// # Safety
///
/// `inputs_json` and `params_json` must be valid null-terminated UTF-8
/// strings allocated and owned by the caller.  Null pointers are
/// accepted and will gracefully produce a null return.
///
/// # Allocation
///
/// If a result were returned, the caller would need to free it via
/// `free_string`.  Since we return null, no allocation occurs.
extern "C" fn infer_shape_fn(
    inputs_json: *const c_char,
    params_json: *const c_char,
) -> *mut c_char {
    // SAFETY: We only read the C strings to silence unused-variable warnings.
    // We don't actually need the values since we return null, but verifying
    // they're non-null helps catch obvious caller mistakes.
    let _inputs = unsafe {
        if inputs_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(inputs_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let _params = unsafe {
        if params_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(params_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    // This plugin bundles 43 block types.  The PluginFFI interface does
    // not provide a block-type parameter, so per-block shape inference
    // is not available through this path.  Return null (error).
    //
    // External consumers should call mlmd_core::block::registry::lookup_block()
    // directly after loading this library.
    std::ptr::null_mut()
}

/// Compute the total parameter count.
///
/// Like `infer_shape_fn`, this requires a block-type discriminator and
/// is not applicable through the generic plugin interface.
///
/// Returns `null` to indicate the operation is not applicable.
///
/// # Safety
///
/// Same safety contract as `infer_shape_fn`.
extern "C" fn param_count_fn(
    inputs_json: *const c_char,
    params_json: *const c_char,
) -> *mut c_char {
    // SAFETY: Same as infer_shape_fn — we only read the C strings to
    // validate them, then return null.
    let _inputs = unsafe {
        if inputs_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(inputs_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let _params = unsafe {
        if params_json.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(params_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    // Not applicable without a block-type discriminator.
    std::ptr::null_mut()
}

/// Generate code for a block.
///
/// Like the other dispatch functions, this requires a block-type
/// discriminator and is not applicable through this interface.
///
/// Returns `null` to indicate the operation is not applicable.
///
/// # Safety
///
/// All four pointer arguments must be valid null-terminated UTF-8
/// strings (or null, which is handled gracefully).
extern "C" fn codegen_fn(
    _target: *const c_char,
    _block_json: *const c_char,
    _input_vars_json: *const c_char,
    _output_vars_json: *const c_char,
) -> *mut c_char {
    std::ptr::null_mut()
}

/// Free a string previously allocated by one of the plugin functions.
///
/// # Safety
///
/// `s` must be a pointer previously returned by `to_c_string` (or any
/// function in this module that allocates a CString) that has not already
/// been freed.
///
/// Passing null is a no-op.
extern "C" fn free_string_fn(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: We trust the caller to provide a pointer that was
        // returned by a previous call to `to_c_string` (or equivalent)
        // from this library.
        unsafe {
            free_rust_string(s);
        }
    }
}
