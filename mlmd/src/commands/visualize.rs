use clap::Args;

use mlmd_core::block::registry::build_block_registry;
use mlmd_core::parser::graph_builder::build_graph;
use mlmd_core::parser::parser::parse;
use mlmd_core::parser::tokenizer::tokenize;
use mlmd_core::shape::infer::infer_shapes;
use mlmd_core::visualize::layout::layout;
use mlmd_core::visualize::renderer::render;

#[derive(Args)]
pub struct VisualizeArgs {
    /// Input .mlmd file
    pub file: String,
    /// Output SVG file path
    #[arg(short, long)]
    pub output: Option<String>,
    /// Output format
    #[arg(short, long, default_value = "svg")]
    pub format: String,
}

pub fn run(args: VisualizeArgs) -> anyhow::Result<()> {
    let source = std::fs::read_to_string(&args.file)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", args.file, e))?;

    let tokens = tokenize(&source);
    let parse_result = parse(&tokens);

    // Report parse errors but continue if we have nodes
    let has_parse_errors = !parse_result.errors.is_empty();
    for err in &parse_result.errors {
        eprintln!(
            "Parse error at {}:{}: {}",
            err.loc.line, err.loc.col, err.message
        );
    }

    if parse_result.nodes.is_empty() && has_parse_errors {
        anyhow::bail!("No nodes parsed, cannot proceed with visualization");
    }

    let graph = build_graph(&parse_result.nodes);
    let registry = build_block_registry();
    let shape_result = infer_shapes(&graph, &registry);

    for err in &shape_result.errors {
        eprintln!(
            "Shape error in block '{}': {}",
            err.block_id, err.message
        );
    }

    if shape_result.graph.blocks.is_empty() {
        anyhow::bail!("Graph has no blocks after shape inference");
    }

    let layout_result = layout(&shape_result.graph);
    let svg = render(&shape_result.graph, &layout_result);

    if let Some(path) = args.output {
        std::fs::write(&path, &svg)
            .map_err(|e| anyhow::anyhow!("Failed to write {}: {}", path, e))?;
        eprintln!("SVG written to {}", path);
    } else {
        print!("{}", svg);
    }

    Ok(())
}
