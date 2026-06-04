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
