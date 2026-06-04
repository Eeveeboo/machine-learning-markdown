use clap::{Args, Subcommand};

use mlmd_core::plugin::scaffold::{generate_plugin_file, to_snake_case};

#[derive(Args)]
pub struct PluginArgs {
    #[command(subcommand)]
    pub action: PluginAction,
}

#[derive(Subcommand)]
pub enum PluginAction {
    /// Create a new block plugin scaffold
    New {
        /// PascalCase block name (e.g. MyCustomLayer)
        block_name: String,
    },
}

pub fn run(args: PluginArgs) -> anyhow::Result<()> {
    match args.action {
        PluginAction::New { block_name } => create_plugin(&block_name),
    }
}

fn create_plugin(block_name: &str) -> anyhow::Result<()> {
    let code = generate_plugin_file(block_name);
    let snake_name = to_snake_case(block_name);
    let filename = format!("{}.rs", snake_name);

    // Check if file already exists
    if std::path::Path::new(&filename).exists() {
        anyhow::bail!(
            "File '{}' already exists. Remove it first or choose a different block name.",
            filename
        );
    }

    std::fs::write(&filename, &code)?;
    println!("Created plugin file: {}", filename);
    Ok(())
}
