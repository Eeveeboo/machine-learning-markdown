use zed_extension_api::{self as zed, node_binary_path, Result};

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
        let node = node_binary_path()?;
        let mlmd_script = format!("{}/dist/bin/mlmd.js", worktree.root_path());

        let env = worktree.shell_env();

        Ok(zed::Command {
            command: node,
            args: vec![mlmd_script, "lsp".to_string(), "--stdio".to_string()],
            env,
        })
    }
}

zed::register_extension!(MlmdExtension);
