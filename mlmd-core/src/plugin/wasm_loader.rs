// ---------------------------------------------------------------------------
// wasm_loader.rs — loads and wraps WASM-based plugins via wasmtime.
//
// WASM plugins export individual named functions (rather than a PluginFFI
// struct with function pointers).  The host uses these exports to call into
// the plugin, passing strings through the WASM linear memory.
//
// WASM plugin exports (all `extern "C" fn` returning/accepting i32):
//
//   plugin_name() -> ptr              — static block type name
//   plugin_params_json() -> ptr       — ParamSpec JSON (caller frees)
//   plugin_infer_shape(i32,i32)->i32  — (inputs_json, params_json) → result
//   plugin_param_count(i32,i32)->i32  — (inputs_json, params_json) → count
//   plugin_codegen(i32,i32,i32,i32)->i32 — (target,block,input_vars,output_vars)
//   plugin_free_string(ptr)           — free a string returned by the plugin
//   plugin_alloc(i32) -> i32          — allocate n bytes, returns pointer
//
// Each i32 "pointer" is an offset into the WASM module's linear memory
// where a null-terminated UTF-8 string resides.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::path::Path;

use wasmtime::{Engine, Linker, Memory, Module, Store, TypedFunc};

use crate::ast::graph::{Block, Shape};
use crate::ast::nodes::ParamValue;
use crate::codegen::result::BlockCodegenResult;

// ---------------------------------------------------------------------------
// Typed function references (cached after instantiation)
// ---------------------------------------------------------------------------

struct WasmFuncs {
    name: TypedFunc<(), i32>,
    params_json: TypedFunc<(), i32>,
    infer_shape: TypedFunc<(i32, i32), i32>,
    param_count: TypedFunc<(i32, i32), i32>,
    codegen: TypedFunc<(i32, i32, i32, i32), i32>,
    free_string: TypedFunc<(i32,), ()>,
    alloc: TypedFunc<(i32,), i32>,
}

// ---------------------------------------------------------------------------
// WasmPlugin
// ---------------------------------------------------------------------------

/// A dynamically loaded WASM plugin backed by a `.wasm` module.
pub struct WasmPlugin {
    store: Store<()>,
    memory: Memory,
    funcs: WasmFuncs,
}

impl WasmPlugin {
    /// Load a plugin from a `.wasm` file.
    ///
    /// The module **must** export the functions listed in the module docstring.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let engine = Engine::default();
        let module = Module::from_file(&engine, path.as_ref())
            .map_err(|e| format!("Failed to load WASM module '{}': {e}", path.as_ref().display()))?;

        let mut store = Store::new(&engine, ());
        let linker = Linker::new(&engine);

        // Instantiate — the plugin should have no imports (no WASI needed
        // for pure computation plugins).
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| format!("Failed to instantiate WASM module: {e}"))?;

        // Get the linear memory export
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| "WASM module must export 'memory'".to_string())?;

        // Helper to extract a typed function — inlined on each use below.
        macro_rules! get_func {
            ($name:literal, $params:ty, $ret:ty) => {
                instance
                    .get_typed_func::<$params, $ret>(&mut store, $name)
                    .map_err(|e| format!("WASM plugin missing '{}' export: {e}", $name))?
            };
        }

        let funcs = WasmFuncs {
            name: get_func!("plugin_name", (), i32),
            params_json: get_func!("plugin_params_json", (), i32),
            infer_shape: get_func!("plugin_infer_shape", (i32, i32), i32),
            param_count: get_func!("plugin_param_count", (i32, i32), i32),
            codegen: get_func!("plugin_codegen", (i32, i32, i32, i32), i32),
            free_string: get_func!("plugin_free_string", (i32,), ()),
            alloc: get_func!("plugin_alloc", (i32,), i32),
        };

        Ok(WasmPlugin {
            store,
            memory,
            funcs,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Allocate `size` bytes in WASM linear memory and return the offset.
    fn alloc(&mut self, size: u32) -> Result<u32, String> {
        self.funcs
            .alloc
            .call(&mut self.store, (size as i32,))
            .map(|p| p as u32)
            .map_err(|e| format!("plugin_alloc failed: {e}"))
    }

    /// Write a string + NUL terminator into WASM memory at the given offset.
    fn write_string(&mut self, offset: u32, s: &str) -> Result<(), String> {
        let bytes = s.as_bytes();
        let len = bytes.len();
        self.memory
            .write(&mut self.store, offset as usize, bytes)
            .map_err(|e| format!("failed to write to WASM memory: {e}"))?;
        // Write NUL terminator
        self.memory
            .write(&mut self.store, (offset as usize) + len, &[0u8])
            .map_err(|e| format!("failed to write NUL to WASM memory: {e}"))
    }

    /// Read a NUL-terminated string from WASM memory starting at `offset`.
    fn read_string(&mut self, ptr: u32) -> Result<String, String> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut offset = ptr as usize;
        loop {
            let mut buf = [0u8];
            self.memory
                .read(&self.store, offset, &mut buf)
                .map_err(|_| "memory read failed".to_string())?;
            if buf[0] == 0 {
                break;
            }
            bytes.push(buf[0]);
            offset += 1;
        }
        String::from_utf8(bytes).map_err(|e| format!("invalid UTF-8 in WASM result: {e}"))
    }

    /// Write a string into WASM memory (allocating space), call a function
    /// that takes NUL-terminated string pointers, then read and free the result.
    #[allow(dead_code)]
    fn call_with_strings<F>(
        &mut self,
        inputs: &[&str],
        call_fn: F,
    ) -> Result<String, String>
    where
        F: FnOnce(&mut Self, Vec<u32>) -> Result<u32, String>,
    {
        let mut allocs: Vec<(u32, u32)> = Vec::new(); // (offset, size)

        // Allocate and write each input string
        for s in inputs {
            let size = (s.len() + 1) as u32; // +1 for NUL
            let offset = self.alloc(size)?;
            self.write_string(offset, s)?;
            allocs.push((offset, size));
        }

        let offsets: Vec<u32> = allocs.iter().map(|(off, _)| *off).collect();

        // Call the plugin function
        let result_ptr = call_fn(self, offsets)?;

        // Read the result
        let result = if result_ptr != 0 {
            self.read_string(result_ptr)?
        } else {
            String::new()
        };

        // Free the result (if any)
        if result_ptr != 0 {
            let _ = self.funcs.free_string.call(&mut self.store, (result_ptr as i32,));
        }

        // Free input allocations (the plugin's plugin_dealloc — but since we
        // allocated with plugin_alloc, the plugin should free with its own free,
        // not free_string.  plugin_alloc uses std::alloc::alloc, so we need
        // the corresponding dealloc.  For simplicity, we leak input allocations
        // since they are tiny and per-call.  A proper implementation would
        // call a plugin_dealloc export.

        Ok(result)
    }

    /// Call a 0-arg function that returns a string pointer.
    fn call_string_0(&mut self, _name: &str, func: TypedFunc<(), i32>) -> String {
        match func.call(&mut self.store, ()) {
            Ok(ptr) if ptr != 0 => {
                let s = self.read_string(ptr as u32).unwrap_or_default();
                let _ = self.funcs.free_string.call(&mut self.store, (ptr,));
                s
            }
            _ => String::new(),
        }
    }

    /// Call a 2-arg function (infer_shape, param_count).
    fn call_string_2(
        &mut self,
        a: &str,
        b: &str,
        func: TypedFunc<(i32, i32), i32>,
    ) -> Result<String, String> {
        let size_a = (a.len() + 1) as u32;
        let size_b = (b.len() + 1) as u32;
        let off_a = self.alloc(size_a)?;
        let off_b = self.alloc(size_b)?;
        self.write_string(off_a, a)?;
        self.write_string(off_b, b)?;

        let result_ptr = func
            .call(&mut self.store, (off_a as i32, off_b as i32))
            .map_err(|e| format!("{e}"))?;

        let result = if result_ptr != 0 {
            self.read_string(result_ptr as u32)?
        } else {
            return Err("null result".to_string());
        };

        if result_ptr != 0 {
            let _ = self.funcs.free_string.call(&mut self.store, (result_ptr,));
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Public API (mirrors DynamicPlugin)
    // -----------------------------------------------------------------------

    /// Return the block type name advertised by this plugin.
    pub fn name(&mut self) -> String {
        self.call_string_0("name", self.funcs.name)
    }

    /// Get parameter specifications as a JSON string.
    pub fn params_json(&mut self) -> String {
        self.call_string_0("params_json", self.funcs.params_json)
    }

    /// Infer output shapes given input shapes and resolved parameters.
    pub fn infer_shape(
        &mut self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        let inputs_json =
            serde_json::to_string(inputs).map_err(|e| format!("serialize inputs: {e}"))?;
        let params_json =
            serde_json::to_string(params).map_err(|e| format!("serialize params: {e}"))?;

        let result_str = self.call_string_2(&inputs_json, &params_json, self.funcs.infer_shape)?;

        serde_json::from_str(&result_str)
            .map_err(|e| format!("deserialize infer_shape result: {e}"))
    }

    /// Compute total parameter count given input shapes and params.
    pub fn param_count(
        &mut self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        let inputs_json = serde_json::to_string(inputs).unwrap_or_default();
        let params_json = serde_json::to_string(params).unwrap_or_default();

        let result_str = self
            .call_string_2(&inputs_json, &params_json, self.funcs.param_count)
            .ok()?;

        if result_str.is_empty() || result_str == "null" {
            return None;
        }

        serde_json::from_str::<serde_json::Value>(&result_str)
            .ok()
            .and_then(|v| v.as_u64().map(|n| n as usize))
            .or_else(|| result_str.parse::<usize>().ok())
    }

    /// Generate code for this block for a specific target.
    pub fn codegen(
        &mut self,
        target: &str,
        block: &Block,
        input_vars: &[String],
        output_vars: &[String],
    ) -> Option<BlockCodegenResult> {
        let target_json = serde_json::to_string(target).unwrap_or_default();
        // Remove surrounding quotes from JSON string value
        let target_clean = target_json.trim_matches('"').to_string();
        let block_json = serde_json::to_string(block).unwrap_or_default();
        let input_vars_json = serde_json::to_string(input_vars).unwrap_or_default();
        let output_vars_json = serde_json::to_string(output_vars).unwrap_or_default();

        let size_t = (target_clean.len() + 1) as u32;
        let size_b = (block_json.len() + 1) as u32;
        let size_i = (input_vars_json.len() + 1) as u32;
        let size_o = (output_vars_json.len() + 1) as u32;

        let off_t = self.alloc(size_t).ok()?;
        let off_b = self.alloc(size_b).ok()?;
        let off_i = self.alloc(size_i).ok()?;
        let off_o = self.alloc(size_o).ok()?;

        self.write_string(off_t, &target_clean).ok()?;
        self.write_string(off_b, &block_json).ok()?;
        self.write_string(off_i, &input_vars_json).ok()?;
        self.write_string(off_o, &output_vars_json).ok()?;

        let result_ptr = self
            .funcs
            .codegen
            .call(&mut self.store, (off_t as i32, off_b as i32, off_i as i32, off_o as i32))
            .ok()?;

        let result = if result_ptr != 0 {
            self.read_string(result_ptr as u32).ok()
        } else {
            None
        };

        if result_ptr != 0 {
            let _ = self.funcs.free_string.call(&mut self.store, (result_ptr,));
        }

        result.and_then(|r| serde_json::from_str(&r).ok())
    }
}
