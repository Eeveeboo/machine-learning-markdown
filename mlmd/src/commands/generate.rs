use std::path::PathBuf;

use clap::Args;

use mlmd_core::block::registry::build_block_registry;
use mlmd_core::codegen::target::get_target;
use mlmd_core::parser::graph_builder::build_graph;
use mlmd_core::parser::parser::parse;
use mlmd_core::parser::tokenizer::tokenize;
use mlmd_core::shape::infer::infer_shapes;

#[derive(Args)]
pub struct GenerateArgs {
    /// Input .mlmd file
    pub file: String,
    /// Code generation target: pytorch, keras, candle
    #[arg(short, long)]
    pub target: String,
    /// Output directory (defaults to current directory)
    #[arg(short, long)]
    pub output: Option<String>,
}

pub fn run(args: GenerateArgs) -> anyhow::Result<()> {
    let source = std::fs::read_to_string(&args.file)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", args.file, e))?;

    let tokens = tokenize(&source);
    let parse_result = parse(&tokens);

    let has_parse_errors = !parse_result.errors.is_empty();
    for err in &parse_result.errors {
        eprintln!(
            "Parse error at {}:{}: {}",
            err.loc.line, err.loc.col, err.message
        );
    }

    if parse_result.nodes.is_empty() && has_parse_errors {
        anyhow::bail!("No nodes parsed, cannot proceed with code generation");
    }

    let graph = build_graph(&parse_result.nodes);
    let registry = build_block_registry();
    let shape_result = infer_shapes(&graph, &registry);

    for err in &shape_result.errors {
        eprintln!("Shape error in block '{}': {}", err.block_id, err.message);
    }

    let target = get_target(&args.target).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown target '{}'. Available: pytorch, keras, candle",
            args.target
        )
    })?;

    let files = target.generate(&shape_result.graph);

    if files.is_empty() {
        anyhow::bail!("No files generated for target '{}'", args.target);
    }

    let out_dir = args
        .output
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    std::fs::create_dir_all(&out_dir)
        .map_err(|e| anyhow::anyhow!("Failed to create output directory: {}", e))?;

    for file in &files {
        let file_path = out_dir.join(&file.path);
        std::fs::write(&file_path, &file.content)
            .map_err(|e| anyhow::anyhow!("Failed to write {}: {}", file_path.display(), e))?;
        eprintln!("Generated {}", file_path.display());
    }

    Ok(())
}
