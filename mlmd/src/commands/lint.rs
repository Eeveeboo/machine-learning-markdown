use clap::Args;

use mlmd_core::block::registry::build_block_registry;
use mlmd_core::lint::{lint, LintDiagnostic, Severity};
use mlmd_core::parser::graph_builder::build_graph;
use mlmd_core::parser::parser::parse;
use mlmd_core::parser::tokenizer::tokenize;

#[derive(Args)]
pub struct LintArgs {
    /// Input .mlmd file(s)
    pub files: Vec<String>,
    /// Output diagnostics as JSON
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: LintArgs) -> anyhow::Result<()> {
    let registry = build_block_registry();

    let mut all_diags: Vec<(String, Vec<LintDiagnostic>)> = Vec::new();
    let mut exit_code: i32 = 0;

    for file in &args.files {
        let source = std::fs::read_to_string(file)
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", file, e))?;

        let tokens = tokenize(&source);
        let parse_result = parse(&tokens);

        // Report parse errors
        for err in &parse_result.errors {
            eprintln!(
                "ERROR [parse] at {}:{}: {}",
                err.loc.line, err.loc.col, err.message
            );
            exit_code = 1;
        }

        if parse_result.nodes.is_empty() {
            if !parse_result.errors.is_empty() {
                // Nodes are empty due to errors, skip further checks
                continue;
            }
            // Empty file — no diagnostics
            continue;
        }

        let graph = build_graph(&parse_result.nodes);
        let diags = lint(&graph, &registry);

        if diags.is_empty() && parse_result.errors.is_empty() {
            continue;
        }

        all_diags.push((file.clone(), diags));
    }

    if args.json {
        let json_output: Vec<serde_json::Value> = all_diags
            .iter()
            .flat_map(|(file, diags)| {
                diags.iter().map(move |d| {
                    serde_json::json!({
                        "file": file,
                        "severity": match d.severity {
                            Severity::Error => "error",
                            Severity::Warning => "warning",
                        },
                        "rule": d.rule,
                        "line": d.loc.line,
                        "col": d.loc.col,
                        "message": d.message,
                    })
                })
            })
            .collect();

        println!("{}", serde_json::to_string_pretty(&json_output)?);
    } else {
        for (_file, diags) in &all_diags {
            for d in diags {
                let severity_str = match d.severity {
                    Severity::Error => "ERROR",
                    Severity::Warning => "WARNING",
                };
                println!(
                    "{} [{}] at {}:{}: {}",
                    severity_str, d.rule, d.loc.line, d.loc.col, d.message
                );
                if d.severity == Severity::Error {
                    exit_code = 1;
                }
            }
        }
    }

    if exit_code != 0 {
        std::process::exit(exit_code);
    }

    Ok(())
}
