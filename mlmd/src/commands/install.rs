use std::process::Command;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct InstallArgs {
    #[command(subcommand)]
    pub editor: InstallEditor,
}

#[derive(Subcommand)]
pub enum InstallEditor {
    /// Install VS Code extension
    Vscode,
    /// Install Zed extension
    Zed,
}

pub fn run(args: InstallArgs) -> anyhow::Result<()> {
    match args.editor {
        InstallEditor::Vscode => install_vscode(),
        InstallEditor::Zed => install_zed(),
    }
}

fn install_vscode() -> anyhow::Result<()> {
    let project_root = std::env::current_dir()?;

    // Source: mlmd-vscode/ in project root
    let source_dir = project_root.join("mlmd-vscode");
    if !source_dir.exists() {
        anyhow::bail!(
            "mlmd-vscode directory not found at {}. Ensure the VS Code extension exists.",
            source_dir.display()
        );
    }

    // Target: ~/.vscode/extensions/mlmd-vscode/
    let home =
        std::env::var("HOME").map_err(|_| anyhow::anyhow!("Cannot determine HOME directory"))?;
    let target_dir = std::path::Path::new(&home)
        .join(".vscode")
        .join("extensions")
        .join("mlmd-vscode");

    // Remove target if it exists, then copy
    if target_dir.exists() {
        eprintln!("Removing existing installation at {}", target_dir.display());
        std::fs::remove_dir_all(&target_dir)?;
    }

    // Create parent directories
    std::fs::create_dir_all(target_dir.parent().unwrap())?;

    // Copy recursively
    eprintln!(
        "Copying {} to {}",
        source_dir.display(),
        target_dir.display()
    );
    copy_dir_recursive(&source_dir, &target_dir)?;

    // Run npm install && npm run build
    eprintln!("Running npm install in {}...", target_dir.display());
    let npm_install = Command::new("npm")
        .args(["install"])
        .current_dir(&target_dir)
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run npm install: {}", e))?;

    if !npm_install.success() {
        anyhow::bail!("npm install failed");
    }

    eprintln!("Running npm run build in {}...", target_dir.display());
    let npm_build = Command::new("npm")
        .args(["run", "build"])
        .current_dir(&target_dir)
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run npm run build: {}", e))?;

    if !npm_build.success() {
        anyhow::bail!("npm run build failed");
    }

    eprintln!(
        "✅ VS Code extension installed successfully at {}",
        target_dir.display()
    );
    eprintln!("Restart VS Code to activate the extension.");
    Ok(())
}

fn install_zed() -> anyhow::Result<()> {
    let project_root = std::env::current_dir()?;

    // Build WASM
    eprintln!("Building mlmd-extension-zed for wasm32-wasip2...");
    let status = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "--release",
            "-p",
            "mlmd-extension-zed",
        ])
        .current_dir(&project_root)
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run cargo build: {}", e))?;

    if !status.success() {
        anyhow::bail!("cargo build failed for mlmd-extension-zed");
    }

    // Copy extension.wasm to project root
    let wasm_source = project_root
        .join("target")
        .join("wasm32-wasip2")
        .join("release")
        .join("mlmd_extension_zed.wasm");

    if !wasm_source.exists() {
        anyhow::bail!(
            "WASM binary not found at {}. Build may have failed.",
            wasm_source.display()
        );
    }

    let wasm_target = project_root.join("extension.wasm");
    std::fs::copy(&wasm_source, &wasm_target)
        .map_err(|e| anyhow::anyhow!("Failed to copy WASM binary: {}", e))?;
    eprintln!(
        "Copied {} → {}",
        wasm_source.display(),
        wasm_target.display()
    );

    // Update extension.toml grammar URL
    let extension_toml_path = project_root.join("extension.toml");
    let toml_content = std::fs::read_to_string(&extension_toml_path)
        .map_err(|e| anyhow::anyhow!("Failed to read extension.toml: {}", e))?;

    let grammar_abs_path = project_root.join("grammars").join("mlmd-grammar");
    let file_url = format!("file://{}", grammar_abs_path.display());

    // Update the grammar URL in extension.toml
    let updated = update_grammar_url(&toml_content, &file_url);

    std::fs::write(&extension_toml_path, &updated)
        .map_err(|e| anyhow::anyhow!("Failed to write extension.toml: {}", e))?;
    eprintln!("Updated extension.toml grammar URL to {}", file_url);

    eprintln!("✅ Zed extension built and configured successfully.");
    Ok(())
}

/// Replace the grammar URL in extension.toml content with the given file URL.
fn update_grammar_url(content: &str, file_url: &str) -> String {
    // Look for the grammar URL line and replace it
    let mut found = false;
    let mut result: Vec<String> = content
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("grammar")
                || (line.trim_start().starts_with("grammars") || line.contains("grammar"))
            {
                // Check if this line contains a URL
                if line.contains("http")
                    || line.contains("file://")
                    || line.contains("grammars/mlmd")
                {
                    found = true;
                    // Replace the URL in this line
                    if let Some(_eq_idx) = line.find('=') {
                        let indent = &line[..line.len() - line.trim_start().len()];
                        format!("{}grammar = \"{}\"", indent, file_url)
                    } else {
                        line.to_string()
                    }
                } else {
                    line.to_string()
                }
            } else {
                line.to_string()
            }
        })
        .collect();

    if !found {
        result.push(format!(
            "\n# grammar URL updated by install command\ngrammar = \"{}\"",
            file_url
        ));
    }

    result.join("\n")
}

/// Recursively copy a directory.
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    if !dst.exists() {
        std::fs::create_dir_all(dst)?;
    }

    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}
