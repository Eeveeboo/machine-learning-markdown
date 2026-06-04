use std::path::Path;

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
        let env = worktree.shell_env();

        // For dev extensions, find mlmd via PATH (npm global install).
        // Fall back to the worktree root (extension == project).
        let mlmd_bin = worktree.which("mlmd").unwrap_or_else(|| {
            let root = worktree.root_path();
            Path::new(&root)
                .join("dist/bin/mlmd.js")
                .to_string_lossy()
                .to_string()
        });

        Ok(zed::Command {
            command: node,
            args: vec![mlmd_bin, "lsp".to_string(), "--stdio".to_string()],
            env,
        })
    }
}

zed::register_extension!(MlmdExtension);
