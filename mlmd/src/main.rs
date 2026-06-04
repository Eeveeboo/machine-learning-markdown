use clap::{Parser, Subcommand};

mod commands;

/// MLMD — Neural Network Architecture DSL
#[derive(Parser)]
#[command(name = "mlmd", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Visualize a .mlmd model as SVG
    Visualize(commands::visualize::VisualizeArgs),
    /// Generate code from a .mlmd model
    Generate(commands::generate::GenerateArgs),
    /// Lint a .mlmd file for errors
    Lint(commands::lint::LintArgs),
    /// Start the LSP server
    Lsp(commands::lsp::LspArgs),
    /// Install editor extensions
    Install(commands::install::InstallArgs),
    /// Plugin management commands
    Plugin(commands::plugin::PluginArgs),
}

fn main() -> anyhow::Result<()> {
    // Register all builtin blocks at startup
    mlmd_builtin_plugins::register_all();
    // Register codegen targets (pytorch, keras, candle)
    mlmd_core::codegen::targets::register_all_codegen_targets();
    // Load dynamic plugins from .mlmdrc (if present)
    load_dynamic_plugins()?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Visualize(args) => commands::visualize::run(args),
        Commands::Generate(args) => commands::generate::run(args),
        Commands::Lint(args) => commands::lint::run(args),
        Commands::Lsp(_args) => commands::lsp::run(),
        Commands::Install(args) => commands::install::run(args),
        Commands::Plugin(args) => commands::plugin::run(args),
    }
}

/// Load dynamic plugins referenced in `.mlmdrc`.
///
/// The `.mlmdrc` configuration file may contain a `"plugins"` field pointing
/// to a `.so` / `.dylib` or `.wasm` file (or an array of such paths).
/// Each plugin is loaded and registered with the global BlockDef and codegen
/// registries via the appropriate adapter method.
fn load_dynamic_plugins() -> anyhow::Result<()> {
    let config = mlmd_core::config::loader::load_config(None)
        .map_err(|e| anyhow::anyhow!("Failed to load .mlmdrc: {e}"))?;

    if let Some(config) = config {
        if let Some(plugin_paths) = &config.plugins {
            let paths: Vec<&str> = plugin_paths.as_paths();

            if paths.is_empty() {
                return Ok(());
            }

            for plugin_path in paths {
                let path_display = plugin_path;

                // Auto-detect WASM vs native based on extension
                if path_display.ends_with(".wasm") {
                    #[cfg(feature = "wasm-plugins")]
                    {
                        let name =
                            mlmd_core::plugin::adapter::register_wasm_plugin(plugin_path)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "Failed to load WASM plugin '{plugin_path}': {e}"
                                    )
                                })?;
                        eprintln!("Loaded WASM plugin: {name} ({plugin_path})");
                    }
                    #[cfg(not(feature = "wasm-plugins"))]
                    {
                        anyhow::bail!(
                            "WASM plugins are not supported in this build. \
                             Rebuild with the 'wasm-plugins' feature enabled. \
                             Path: {plugin_path}"
                        );
                    }
                } else {
                    #[cfg(feature = "dynamic-plugins")]
                    {
                        let name =
                            mlmd_core::plugin::adapter::register_dynamic_plugin(plugin_path)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "Failed to load plugin '{plugin_path}': {e}"
                                    )
                                })?;
                        eprintln!("Loaded dynamic plugin: {name} ({plugin_path})");
                    }
                    #[cfg(not(feature = "dynamic-plugins"))]
                    {
                        anyhow::bail!(
                            "Dynamic plugins are not supported in this build. \
                             Rebuild with the 'dynamic-plugins' feature enabled."
                        );
                    }
                }
            }
        }
    }

    Ok(())
}
