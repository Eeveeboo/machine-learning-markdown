// ---------------------------------------------------------------------------
// mlmd-plugin-api-macros — proc-macro crate for mlmd-plugin-api.
//
// Provides the `register_plugin!` macro that reduces plugin registration
// boilerplate to a single macro invocation.
// ---------------------------------------------------------------------------

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Ident};

// ---------------------------------------------------------------------------
// register_plugin!
// ---------------------------------------------------------------------------

/// Register a plugin type with the NNML plugin system.
///
/// Given a type that implements the `Plugin` trait, this macro generates:
///
/// - A `BlockDef` adapter struct wrapping your type
/// - Codegen dispatch functions for pytorch, keras, and candle targets
/// - A `pub fn register()` function that wires everything into the global
///   registries
///
/// # Usage
///
/// ```ignore
/// use mlmd_plugin_api::*;
///
/// struct MyBlock;
///
/// impl Plugin for MyBlock {
///     // ... methods ...
/// }
///
/// register_plugin!(MyBlock);
/// ```
///
/// The generated `register()` function is then called from your
/// crate's top-level registration.
#[proc_macro]
pub fn register_plugin(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Ident);
    let ty = &input;
    let adapter_ident = Ident::new(&format!("{}PluginAdapter", ty), ty.span());
    let pytorch_fn = Ident::new(
        &format!("__{}_pytorch_codegen", to_snake(&ty.to_string())),
        ty.span(),
    );
    let keras_fn = Ident::new(
        &format!("__{}_keras_codegen", to_snake(&ty.to_string())),
        ty.span(),
    );
    let candle_fn = Ident::new(
        &format!("__{}_candle_codegen", to_snake(&ty.to_string())),
        ty.span(),
    );

    let expanded = quote! {
        // -----------------------------------------------------------------------
        // BlockDef adapter — wraps the Plugin impl for the shape inference pipeline
        // -----------------------------------------------------------------------

        struct #adapter_ident {
            inner: #ty,
            param_cache: ::std::vec::Vec<::mlmd_core::block::types::ParamSpec>,
        }

        // SAFETY: #ty implements Plugin (Send + Sync), and ParamSpec is Send + Sync.
        unsafe impl ::std::marker::Send for #adapter_ident {}
        unsafe impl ::std::marker::Sync for #adapter_ident {}

        impl ::mlmd_core::block::types::BlockDef for #adapter_ident {
            fn name(&self) -> &str {
                self.inner.name()
            }

            fn params(&self) -> &[::mlmd_core::block::types::ParamSpec] {
                &self.param_cache
            }

            fn infer_shape(
                &self,
                inputs: &[::mlmd_core::ast::graph::Shape],
                params: &::std::collections::HashMap<
                    ::std::string::String,
                    ::mlmd_core::ast::nodes::ParamValue,
                >,
            ) -> ::std::result::Result<
                ::std::vec::Vec<::mlmd_core::ast::graph::Shape>,
                ::std::string::String,
            > {
                self.inner.infer_shape(inputs, params)
            }

            fn param_count(
                &self,
                inputs: &[::mlmd_core::ast::graph::Shape],
                params: &::std::collections::HashMap<
                    ::std::string::String,
                    ::mlmd_core::ast::nodes::ParamValue,
                >,
            ) -> ::std::option::Option<::std::primitive::usize> {
                self.inner.param_count(inputs, params)
            }

            fn show_depth(&self) -> bool {
                self.inner.show_depth()
            }

            fn num_inputs(&self) -> Option<usize> {
                self.inner.num_inputs()
            }

            fn num_outputs(&self) -> Option<usize> {
                self.inner.num_outputs()
            }
        }

        // -----------------------------------------------------------------------
        // Codegen dispatch functions — one per target
        // -----------------------------------------------------------------------

        fn #pytorch_fn(
            block: &::mlmd_core::ast::graph::Block,
            input_vars: &[::std::string::String],
            output_vars: &[::std::string::String],
        ) -> ::mlmd_core::plugin::traits::BlockCodegenResult {
            let plugin = #ty;
            plugin.codegen("pytorch", block, input_vars, output_vars)
                .expect(concat!(
                    stringify!(#ty),
                    " plugin: codegen for target 'pytorch' returned None; ",
                    "a Plugin::codegen() implementation must return Some(...) for each supported target"
                ))
        }

        fn #keras_fn(
            block: &::mlmd_core::ast::graph::Block,
            input_vars: &[::std::string::String],
            output_vars: &[::std::string::String],
        ) -> ::mlmd_core::plugin::traits::BlockCodegenResult {
            let plugin = #ty;
            plugin.codegen("keras", block, input_vars, output_vars)
                .expect(concat!(
                    stringify!(#ty),
                    " plugin: codegen for target 'keras' returned None"
                ))
        }

        fn #candle_fn(
            block: &::mlmd_core::ast::graph::Block,
            input_vars: &[::std::string::String],
            output_vars: &[::std::string::String],
        ) -> ::mlmd_core::plugin::traits::BlockCodegenResult {
            let plugin = #ty;
            plugin.codegen("candle", block, input_vars, output_vars)
                .expect(concat!(
                    stringify!(#ty),
                    " plugin: codegen for target 'candle' returned None"
                ))
        }

        // -----------------------------------------------------------------------
        // Registration entry point
        // -----------------------------------------------------------------------

        /// Register this plugin type with the global registries.
        ///
        /// Called automatically by the crate-level `register_all()` or
        /// directly by the user.
        pub fn register() {
            let plugin = #ty;
            let name_: ::std::string::String = plugin.name().to_string();
            let params_: ::std::vec::Vec<::mlmd_core::block::types::ParamSpec> = plugin.params();
            let adapter = #adapter_ident {
                inner: plugin,
                param_cache: params_,
            };
            ::mlmd_core::block::registry::register_block(::std::boxed::Box::new(adapter));
            ::mlmd_core::plugin::registry::register_block_codegen(&name_, "pytorch", #pytorch_fn);
            ::mlmd_core::plugin::registry::register_block_codegen(&name_, "keras", #keras_fn);
            ::mlmd_core::plugin::registry::register_block_codegen(&name_, "candle", #candle_fn);
        }
    };

    TokenStream::from(expanded)
}

// ---------------------------------------------------------------------------
// export_plugin!
// ---------------------------------------------------------------------------

/// Export the `MLMD_PLUGIN` C-ABI symbol for dynamic loading.
///
/// Generates all the FFI boilerplate (`PluginFFI` table, 6 extern "C"
/// function implementations, and the `MLMD_PLUGIN` entry point) for a
/// plugin type previously registered with `register_plugin!`.
///
/// After this macro, your plugin `.so`/`.dylib` can be loaded via
/// `DynamicPlugin::load`.
///
/// # Usage
///
/// ```ignore
/// use mlmd_plugin_api::*;
///
/// struct MyBlock;
/// impl Plugin for MyBlock { ... }
/// register_plugin!(MyBlock);
///
/// // Optional: export the FFI symbol for dynamic loading
/// export_plugin!(MyBlock);
/// ```
#[proc_macro]
pub fn export_plugin(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Ident);
    let ty = &input;

    let expanded = quote! {
        // -----------------------------------------------------------------------
        // FFI export for #ty — generated by export_plugin!
        //
        // Provides the MLMD_PLUGIN C-ABI symbol that DynamicPlugin::load
        // discovers via libloading.
        // -----------------------------------------------------------------------

        use ::std::ffi::CStr;
        use ::std::os::raw::c_char;
        use ::std::sync::OnceLock;
        use ::mlmd_core::plugin::ffi::{
            codegen_result_to_json, free_rust_string, parse_block_json, parse_params_json,
            parse_shapes_json, parse_string_list_json, shapes_to_json, specs_to_json,
            to_c_string, PluginFFI,
        };

        /// Cached FFI function table, initialised on first access.
        static __FFI: OnceLock<PluginFFI> = OnceLock::new();

        fn __get_ffi() -> &'static PluginFFI {
            __FFI.get_or_init(|| {
                // Ensure the plugin is registered with the static registries
                // before exposing the FFI table.
                register();

                PluginFFI {
                    name: __name_fn,
                    params_json: __params_json_fn,
                    infer_shape: __infer_shape_fn,
                    param_count: __param_count_fn,
                    codegen: __codegen_fn,
                    free_string: __free_string_fn,
                }
            })
        }

        /// Entry point discovered by `DynamicPlugin::load` via `libloading`.
        ///
        /// Returns a pointer to a static `PluginFFI` that lives for the
        /// duration of the loaded library.
        ///
        /// # Safety
        ///
        /// The returned pointer is valid for the lifetime of the loaded library.
        #[no_mangle]
        pub extern "C" fn MLMD_PLUGIN() -> *const PluginFFI {
            __get_ffi() as *const PluginFFI
        }

        /// Return the block type name as a static C string.
        extern "C" fn __name_fn() -> *const c_char {
            static NAME: OnceLock<::std::ffi::CString> = OnceLock::new();
            NAME.get_or_init(|| {
                let plugin = #ty;
                ::std::ffi::CString::new(plugin.name()).unwrap()
            })
            .as_ptr()
        }

        /// Return parameter specifications as a JSON string.
        ///
        /// The caller must free the returned string via the `free_string`
        /// function pointer.
        extern "C" fn __params_json_fn() -> *mut c_char {
            let plugin = #ty;
            let params = plugin.params();
            to_c_string(&specs_to_json(&params))
        }

        /// Infer output shapes from serialised input shapes and params.
        ///
        /// # Safety
        ///
        /// Both `inputs_json` and `params_json` must be valid null-terminated
        /// UTF-8 strings.  Null pointers are handled gracefully.
        extern "C" fn __infer_shape_fn(
            inputs_json: *const c_char,
            params_json: *const c_char,
        ) -> *mut c_char {
            let inputs_str = unsafe {
                if inputs_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match CStr::from_ptr(inputs_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let params_str = unsafe {
                if params_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match CStr::from_ptr(params_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };

            let inputs = match parse_shapes_json(inputs_str) {
                Ok(v) => v,
                Err(_) => return ::std::ptr::null_mut(),
            };
            let params = match parse_params_json(params_str) {
                Ok(p) => p,
                Err(_) => return ::std::ptr::null_mut(),
            };

            if inputs.is_empty() {
                return ::std::ptr::null_mut();
            }

            let plugin = #ty;
            match plugin.infer_shape(&inputs, &params) {
                Ok(shapes) => to_c_string(&shapes_to_json(&shapes)),
                Err(_) => ::std::ptr::null_mut(),
            }
        }

        /// Compute the total number of learnable parameters.
        ///
        /// # Safety
        ///
        /// Same contract as `__infer_shape_fn`.
        extern "C" fn __param_count_fn(
            inputs_json: *const c_char,
            params_json: *const c_char,
        ) -> *mut c_char {
            let inputs_str = unsafe {
                if inputs_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match CStr::from_ptr(inputs_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let params_str = unsafe {
                if params_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match CStr::from_ptr(params_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };

            let inputs = match parse_shapes_json(inputs_str) {
                Ok(v) => v,
                Err(_) => return ::std::ptr::null_mut(),
            };
            let params = match parse_params_json(params_str) {
                Ok(p) => p,
                Err(_) => return ::std::ptr::null_mut(),
            };

            let plugin = #ty;
            match plugin.param_count(&inputs, &params) {
                Some(count) => to_c_string(&count.to_string()),
                None => ::std::ptr::null_mut(),
            }
        }

        /// Generate code for the specified target.
        ///
        /// # Safety
        ///
        /// All four pointer arguments must be valid null-terminated UTF-8
        /// strings.  Null pointers are handled gracefully.
        extern "C" fn __codegen_fn(
            target: *const c_char,
            block_json: *const c_char,
            input_vars_json: *const c_char,
            output_vars_json: *const c_char,
        ) -> *mut c_char {
            let target_str = unsafe {
                if target.is_null() {
                    return ::std::ptr::null_mut();
                }
                match CStr::from_ptr(target).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let block_str = unsafe {
                if block_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match CStr::from_ptr(block_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let input_vars_str = unsafe {
                if input_vars_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match CStr::from_ptr(input_vars_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let output_vars_str = unsafe {
                if output_vars_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match CStr::from_ptr(output_vars_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };

            let block = match parse_block_json(block_str) {
                Ok(b) => b,
                Err(_) => return ::std::ptr::null_mut(),
            };
            let input_vars = match parse_string_list_json(input_vars_str) {
                Ok(v) => v,
                Err(_) => return ::std::ptr::null_mut(),
            };
            let output_vars = match parse_string_list_json(output_vars_str) {
                Ok(v) => v,
                Err(_) => return ::std::ptr::null_mut(),
            };

            let plugin = #ty;
            match plugin.codegen(target_str, &block, &input_vars, &output_vars) {
                Some(result) => to_c_string(&codegen_result_to_json(&result)),
                None => ::std::ptr::null_mut(),
            }
        }

        /// Free a string previously allocated by one of the FFI functions.
        ///
        /// # Safety
        ///
        /// `s` must be a pointer previously returned by `to_c_string` that has
        /// not already been freed.  Null pointers are handled gracefully.
        extern "C" fn __free_string_fn(s: *mut c_char) {
            if !s.is_null() {
                // SAFETY: We trust the caller to provide a valid pointer.
                unsafe {
                    free_rust_string(s);
                }
            }
        }
    };

    TokenStream::from(expanded)
}

// ---------------------------------------------------------------------------
// export_wasm_plugin!
// ---------------------------------------------------------------------------

/// Export individual WASM-compatible functions for loading as a `.wasm` plugin.
///
/// Generates `#[no_mangle] extern "C"` functions that the WASM host discovers
/// by name.  Each function has the same signature as the native C-ABI version
/// but is exported as an individual symbol rather than through a `PluginFFI`
/// struct.
///
/// Generated exports:
///   - `plugin_name`              — returns pointer to block type name
///   - `plugin_params_json`       — returns pointer to param specs JSON
///   - `plugin_infer_shape(i32,i32) -> i32`
///   - `plugin_param_count(i32,i32) -> i32`
///   - `plugin_codegen(i32,i32,i32,i32) -> i32`
///   - `plugin_free_string(i32)`  — free a returned string
///   - `plugin_alloc(i32) -> i32` — allocate n bytes, returns pointer
///
/// # Usage
///
/// ```ignore
/// use mlmd_plugin_api::*;
///
/// struct MyBlock;
/// impl Plugin for MyBlock { ... }
/// register_plugin!(MyBlock);
///
/// // For WASM dynamic loading:
/// export_wasm_plugin!(MyBlock);
/// ```
#[proc_macro]
pub fn export_wasm_plugin(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Ident);
    let ty = &input;

    let expanded = quote! {
        // -----------------------------------------------------------------------
        // WASM exports for #ty — generated by export_wasm_plugin!
        //
        // Each function follows the same C-ABI convention as the native
        // export_plugin! macro, but is exported as an individual `#[no_mangle]`
        // symbol for discovery by the WASM host.
        //
        // NOTE: All paths are fully qualified to avoid conflicts when
        // export_plugin! and export_wasm_plugin! are used together.
        // -----------------------------------------------------------------------

        /// Return the block type name as a static C string.
        #[no_mangle]
        pub extern "C" fn plugin_name() -> *const ::std::os::raw::c_char {
            static NAME: ::std::sync::OnceLock<::std::ffi::CString> =
                ::std::sync::OnceLock::new();
            NAME.get_or_init(|| {
                let plugin = #ty;
                ::std::ffi::CString::new(plugin.name()).unwrap()
            })
            .as_ptr()
        }

        /// Return parameter specifications as a JSON string.
        ///
        /// The caller must free the returned string via `plugin_free_string`.
        #[no_mangle]
        pub extern "C" fn plugin_params_json() -> *mut ::std::os::raw::c_char {
            let plugin = #ty;
            let params = plugin.params();
            ::mlmd_core::plugin::ffi::to_c_string(
                &::mlmd_core::plugin::ffi::specs_to_json(&params)
            )
        }

        /// Infer output shapes from serialised input shapes and params.
        ///
        /// # Safety
        ///
        /// Both `inputs_json` and `params_json` must be valid null-terminated
        /// UTF-8 strings.  Null pointers are handled gracefully.
        #[no_mangle]
        pub extern "C" fn plugin_infer_shape(
            inputs_json: *const ::std::os::raw::c_char,
            params_json: *const ::std::os::raw::c_char,
        ) -> *mut ::std::os::raw::c_char {
            let inputs_str = unsafe {
                if inputs_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match ::std::ffi::CStr::from_ptr(inputs_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let params_str = unsafe {
                if params_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match ::std::ffi::CStr::from_ptr(params_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };

            let inputs = match ::mlmd_core::plugin::ffi::parse_shapes_json(inputs_str) {
                Ok(v) => v,
                Err(_) => return ::std::ptr::null_mut(),
            };
            let params = match ::mlmd_core::plugin::ffi::parse_params_json(params_str) {
                Ok(p) => p,
                Err(_) => return ::std::ptr::null_mut(),
            };

            if inputs.is_empty() {
                return ::std::ptr::null_mut();
            }

            let plugin = #ty;
            match plugin.infer_shape(&inputs, &params) {
                Ok(shapes) => ::mlmd_core::plugin::ffi::to_c_string(
                    &::mlmd_core::plugin::ffi::shapes_to_json(&shapes)
                ),
                Err(_) => ::std::ptr::null_mut(),
            }
        }

        /// Compute the total number of learnable parameters.
        ///
        /// # Safety
        ///
        /// Same contract as `plugin_infer_shape`.
        #[no_mangle]
        pub extern "C" fn plugin_param_count(
            inputs_json: *const ::std::os::raw::c_char,
            params_json: *const ::std::os::raw::c_char,
        ) -> *mut ::std::os::raw::c_char {
            let inputs_str = unsafe {
                if inputs_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match ::std::ffi::CStr::from_ptr(inputs_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let params_str = unsafe {
                if params_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match ::std::ffi::CStr::from_ptr(params_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };

            let inputs = match ::mlmd_core::plugin::ffi::parse_shapes_json(inputs_str) {
                Ok(v) => v,
                Err(_) => return ::std::ptr::null_mut(),
            };
            let params = match ::mlmd_core::plugin::ffi::parse_params_json(params_str) {
                Ok(p) => p,
                Err(_) => return ::std::ptr::null_mut(),
            };

            let plugin = #ty;
            match plugin.param_count(&inputs, &params) {
                Some(count) => ::mlmd_core::plugin::ffi::to_c_string(&count.to_string()),
                None => ::std::ptr::null_mut(),
            }
        }

        /// Generate code for the specified target.
        ///
        /// # Safety
        ///
        /// All four pointer arguments must be valid null-terminated UTF-8
        /// strings.  Null pointers are handled gracefully.
        #[no_mangle]
        pub extern "C" fn plugin_codegen(
            target: *const ::std::os::raw::c_char,
            block_json: *const ::std::os::raw::c_char,
            input_vars_json: *const ::std::os::raw::c_char,
            output_vars_json: *const ::std::os::raw::c_char,
        ) -> *mut ::std::os::raw::c_char {
            let target_str = unsafe {
                if target.is_null() {
                    return ::std::ptr::null_mut();
                }
                match ::std::ffi::CStr::from_ptr(target).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let block_str = unsafe {
                if block_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match ::std::ffi::CStr::from_ptr(block_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let input_vars_str = unsafe {
                if input_vars_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match ::std::ffi::CStr::from_ptr(input_vars_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };
            let output_vars_str = unsafe {
                if output_vars_json.is_null() {
                    return ::std::ptr::null_mut();
                }
                match ::std::ffi::CStr::from_ptr(output_vars_json).to_str() {
                    Ok(s) => s,
                    Err(_) => return ::std::ptr::null_mut(),
                }
            };

            let block = match ::mlmd_core::plugin::ffi::parse_block_json(block_str) {
                Ok(b) => b,
                Err(_) => return ::std::ptr::null_mut(),
            };
            let input_vars = match ::mlmd_core::plugin::ffi::parse_string_list_json(input_vars_str)
            {
                Ok(v) => v,
                Err(_) => return ::std::ptr::null_mut(),
            };
            let output_vars = match ::mlmd_core::plugin::ffi::parse_string_list_json(
                output_vars_str,
            ) {
                Ok(v) => v,
                Err(_) => return ::std::ptr::null_mut(),
            };

            let plugin = #ty;
            match plugin.codegen(target_str, &block, &input_vars, &output_vars) {
                Some(result) => ::mlmd_core::plugin::ffi::to_c_string(
                    &::mlmd_core::plugin::ffi::codegen_result_to_json(&result),
                ),
                None => ::std::ptr::null_mut(),
            }
        }

        /// Free a string previously allocated by one of the plugin functions.
        ///
        /// # Safety
        ///
        /// `s` must be a pointer previously returned by `to_c_string` that has
        /// not already been freed.  Null pointers are handled gracefully.
        #[no_mangle]
        pub extern "C" fn plugin_free_string(s: *mut ::std::os::raw::c_char) {
            if !s.is_null() {
                unsafe {
                    ::mlmd_core::plugin::ffi::free_rust_string(s);
                }
            }
        }

        /// Allocate `size` bytes and return a pointer.
        ///
        /// This is used by the WASM host to allocate memory for input strings
        /// before calling a plugin function.
        #[no_mangle]
        pub extern "C" fn plugin_alloc(size: i32) -> *mut ::std::os::raw::c_char {
            if size <= 0 {
                return ::std::ptr::null_mut();
            }
            let layout = match ::std::alloc::Layout::from_size_align(size as usize, 1) {
                Ok(l) => l,
                Err(_) => return ::std::ptr::null_mut(),
            };
            unsafe { ::std::alloc::alloc(layout) as *mut ::std::os::raw::c_char }
        }
    };

    TokenStream::from(expanded)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert PascalCase to snake_case for generating identifier names.
fn to_snake(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.char_indices() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            for lower in c.to_lowercase() {
                result.push(lower);
            }
        } else {
            result.push(c);
        }
    }
    result
}
