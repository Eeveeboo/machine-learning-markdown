use zed_extension_api::{self as zed, Result};

struct MlmdExtension;

impl zed::Extension for MlmdExtension {
    fn new() -> Self {
        MlmdExtension
    }

    fn language_server_command(
        &mut self,
        _language_server_id: &zed::LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<zed::Command> {
        let npx = worktree
            .which("npx")
            .map_err(|e| format!("failed to locate npx: {e}"))?
            .ok_or_else(|| "npx not found in PATH — is Node.js installed?".to_string())?;

        Ok(zed::Command {
            command: npx,
            args: vec!["mlmd".to_string(), "lsp".to_string()],
            env: worktree.shell_env(),
        })
    }
}

zed::register_extension!(MlmdExtension);
