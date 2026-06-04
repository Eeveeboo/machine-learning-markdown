use clap::{Args, Subcommand};
use std::collections::HashSet;
use std::path::Path;

use mlmd_core::config::loader::{MlmdConfig, PluginPaths};
use mlmd_core::plugin::scaffold::{
    generate_plugin_cargo_toml, generate_plugin_file, generate_plugin_file_wasm, kebab_to_pascal,
    to_snake_case,
};

#[derive(Args)]
pub struct PluginArgs {
    #[command(subcommand)]
    pub action: PluginAction,
}

#[derive(Subcommand)]
pub enum PluginAction {
    /// Create a new block plugin scaffold (single file)
    New {
        /// PascalCase block name (e.g. MyCustomLayer)
        block_name: String,
    },
    /// List all registered blocks (builtin + dynamic plugins)
    List,
    /// Bootstrap a new plugin project and register it in .mlmdrc
    Init {
        /// Kebab-case plugin package name (e.g. my-cool-plugin)
        plugin_name: String,
        /// PascalCase block name (defaults to PascalCase(plugin_name))
        #[arg(long)]
        block_name: Option<String>,
        /// Target WASM instead of native .so/.dylib
        #[arg(long, default_value_t = false)]
        wasm: bool,
    },
}

pub fn run(args: PluginArgs) -> anyhow::Result<()> {
    match args.action {
        PluginAction::New { block_name } => create_plugin(&block_name),
        PluginAction::List => list_plugins(),
        PluginAction::Init {
            plugin_name,
            block_name,
            wasm,
        } => init_plugin(&plugin_name, block_name.as_deref(), wasm),
    }
}

fn create_plugin(block_name: &str) -> anyhow::Result<()> {
    let code = generate_plugin_file(block_name);
    let snake_name = to_snake_case(block_name);
    let filename = format!("{}.rs", snake_name);

    if Path::new(&filename).exists() {
        anyhow::bail!(
            "File '{}' already exists. Remove it first or choose a different block name.",
            filename
        );
    }

    std::fs::write(&filename, &code)?;
    println!("Created plugin file: {}", filename);
    Ok(())
}

fn list_plugins() -> anyhow::Result<()> {
    let all = mlmd_core::block::registry::all_block_names();

    #[cfg(feature = "dynamic-plugins")]
    let mut dynamic = mlmd_core::plugin::adapter::all_dynamic_plugins();
    #[cfg(not(feature = "dynamic-plugins"))]
    let mut dynamic: Vec<(String, String)> = Vec::new();

    let dynamic_names: HashSet<&str> = dynamic.iter().map(|(n, _)| n.as_str()).collect();
    let mut builtin: Vec<&String> = all
        .iter()
        .filter(|n| !dynamic_names.contains(n.as_str()))
        .collect();

    println!();
    println!(
        "Registered blocks — {} builtin, {} dynamic",
        builtin.len(),
        dynamic.len()
    );
    println!();

    builtin.sort();
    dynamic.sort_by(|a, b| a.0.cmp(&b.0));

    if !builtin.is_empty() {
        println!("Builtin blocks:");
        for name in &builtin {
            println!("  {}", format_block_signature(name));
        }
        println!();
    }

    if !dynamic.is_empty() {
        println!("Dynamic plugins:");
        for (name, path) in &dynamic {
            println!("  {}", format_block_signature(name));
            println!("    → {path}");
        }
        println!();
    }

    Ok(())
}

/// Format a block as `[inputs] -> Name(params) -> [outputs]`.
fn format_block_signature(name: &str) -> String {
    let Some(guard) = mlmd_core::block::registry::lookup_block(name) else {
        return format!("[?] -> {name} -> [?]");
    };

    let inputs = format_io_labels(guard.num_inputs());
    let outputs = format_io_labels(guard.num_outputs());
    let params = format_params(guard.params());

    format!("{inputs} -> {name}{params} -> {outputs}")
}

/// Format a parameter list: `(p1: type, [p2: type])` or empty string.
fn format_params(params: &[mlmd_core::block::types::ParamSpec]) -> String {
    if params.is_empty() {
        return String::new();
    }
    let items: Vec<String> = params.iter().map(format_param).collect();
    format!("({})", items.join(", "))
}

/// Format a single `ParamSpec` as `name: type` (required) or `[name: type]`
/// (optional), appending ` = default` when a default exists.
fn format_param(spec: &mlmd_core::block::types::ParamSpec) -> String {
    let type_str = spec.param_type.as_str();
    if spec.required {
        format!("{}: {}", spec.name, type_str)
    } else {
        match &spec.default {
            Some(val) => format!("[{}: {} = {}]", spec.name, type_str, val),
            None => format!("[{}: {}]", spec.name, type_str),
        }
    }
}

/// Format an input/output label from an `Option<usize>` count.
///
/// - `Some(0)` → `[]`
/// - `Some(1)` → `[x]`
/// - `Some(2)` → `[a, b]`
/// - `Some(n)` → `[a, b, c, …]`
/// - `None`   → `[…]`
fn format_io_labels(count: Option<usize>) -> String {
    const LABELS: &[&str] = &["a", "b", "c", "d", "e", "f"];
    match count {
        Some(0) => "[]".to_string(),
        Some(1) => "[x]".to_string(),
        Some(n) if n <= LABELS.len() => {
            let items: Vec<&str> = LABELS[..n].to_vec();
            format!("[{}]", items.join(", "))
        }
        Some(n) => {
            let items: Vec<&str> = LABELS.iter().copied().take(n - 1).collect();
            format!("[{}, …]", items.join(", "))
        }
        None => "[…]".to_string(),
    }
}

fn init_plugin(plugin_name: &str, block_name: Option<&str>, wasm: bool) -> anyhow::Result<()> {
    let block_name = block_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| kebab_to_pascal(plugin_name));

    let plugin_dir = Path::new(plugin_name);
    if plugin_dir.exists() {
        anyhow::bail!("Directory '{}' already exists.", plugin_name);
    }

    // Create directory structure
    let src_dir = plugin_dir.join("src");
    std::fs::create_dir_all(&src_dir)
        .map_err(|e| anyhow::anyhow!("Failed to create plugin directory: {e}"))?;

    // Write Cargo.toml
    let cargo_toml = generate_plugin_cargo_toml(plugin_name, &block_name);
    std::fs::write(plugin_dir.join("Cargo.toml"), &cargo_toml)
        .map_err(|e| anyhow::anyhow!("Failed to write Cargo.toml: {e}"))?;
    println!("  Created {}/Cargo.toml", plugin_name);

    // Write src/lib.rs
    let lib_rs = if wasm {
        generate_plugin_file_wasm(&block_name)
    } else {
        generate_plugin_file(&block_name)
    };
    std::fs::write(src_dir.join("lib.rs"), &lib_rs)
        .map_err(|e| anyhow::anyhow!("Failed to write src/lib.rs: {e}"))?;
    println!("  Created {}/src/lib.rs", plugin_name);

    // Register in .mlmdrc
    let plugin_path = if wasm {
        format!("target/wasm32-wasi/debug/{}.wasm", plugin_name.replace('-', "_"))
    } else {
        let lib_filename = format!(
            "{}{}{}",
            std::env::consts::DLL_PREFIX,
            plugin_name.replace('-', "_"),
            std::env::consts::DLL_SUFFIX
        );
        format!("target/debug/{}", lib_filename)
    };

    let config = MlmdConfig {
        plugins: Some(PluginPaths::Single(plugin_path)),
        targets: None,
    };

    mlmd_core::config::loader::save_config(&config)
        .map_err(|e| anyhow::anyhow!("Failed to save .mlmdrc: {e}"))?;
    println!("  Updated .mlmdrc");

    println!();
    println!("Next steps:");
    if wasm {
        println!("  cd {plugin_name}");
        println!("  rustup target add wasm32-wasi");
        println!("  cargo build --target wasm32-wasi --release");
        println!("  Then use \"{block_name}\" in your .mlmd files");
    } else {
        println!("  cd {plugin_name} && cargo build");
        println!("  Then use \"{block_name}\" in your .mlmd files");
    }

    Ok(())
}
