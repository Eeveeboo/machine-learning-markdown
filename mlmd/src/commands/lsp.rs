use clap::Args;

#[derive(Args)]
pub struct LspArgs;

pub fn run() -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(mlmd_core::lsp::server::start_lsp_server());
    Ok(())
}
