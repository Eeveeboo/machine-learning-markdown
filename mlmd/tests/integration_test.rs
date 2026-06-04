// ---------------------------------------------------------------------------
// End-to-end integration tests for the MLMD pipeline.
//
// These tests validate the full pipeline: tokenize → parse → build graph →
// shape inference → codegen — using real .mlmd files from the examples/
// directory and the builtin block registry.
// ---------------------------------------------------------------------------

use std::path::PathBuf;

/// Return the absolute path to the workspace root.
fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap().to_path_buf()
}

/// Return the absolute path to an example .mlmd file.
fn example_path(name: &str) -> PathBuf {
    workspace_root().join("examples").join(name)
}

/// Read an example .mlmd file.
fn read_example(name: &str) -> String {
    let path = example_path(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test the full pipeline on the LeNet-5 example.
#[test]
fn test_lenet5_full_pipeline() {
    // 1. Read lenet.mlmd
    let source = read_example("lenet.mlmd");

    // 2. Register builtin blocks and codegen targets
    mlmd_builtin_plugins::register_all();
    mlmd_core::codegen::targets::register_all_codegen_targets();

    // 3. Tokenize
    let tokens = mlmd_core::parser::tokenizer::tokenize(&source);
    assert!(!tokens.is_empty(), "Tokenizer should produce tokens");

    // 4. Parse
    let parse_result = mlmd_core::parser::parser::parse(&tokens);
    assert!(
        parse_result.errors.is_empty(),
        "Parse errors: {:?}",
        parse_result.errors
    );
    assert!(!parse_result.nodes.is_empty(), "Parser should produce nodes");

    // 5. Build graph
    let graph = mlmd_core::parser::graph_builder::build_graph(&parse_result.nodes);
    assert!(!graph.blocks.is_empty(), "Graph should have blocks");
    assert!(!graph.edges.is_empty(), "Graph should have edges");

    // 6. Verify shape inference
    let registry = mlmd_core::block::registry::build_block_registry();
    let shape_result = mlmd_core::shape::infer_shapes(&graph, &registry);
    assert!(
        shape_result.errors.is_empty(),
        "Shape inference errors: {:?}",
        shape_result.errors
    );

    // 7. Verify codegen targets produce output
    let targets = ["pytorch", "candle", "keras"];
    for target_name in &targets {
        let result = mlmd_core::codegen::target::with_target(target_name, |target| {
            target.generate(&shape_result.graph)
        });
        assert!(
            result.is_some(),
            "Codegen target '{}' should exist",
            target_name
        );
        let files = result.unwrap();
        assert!(
            !files.is_empty(),
            "Codegen target '{}' should produce files",
            target_name
        );
    }

    // 8. Structural assertions
    println!(
        "LeNet-5: {} blocks, {} edges, {} groups",
        graph.blocks.len(),
        graph.edges.len(),
        graph.groups.len()
    );
}

/// Test that the tokenizer handles all example files without crashing.
#[test]
fn test_all_examples_tokenize() {
    let examples = ["attention.mlmd", "inception.mlmd", "lenet.mlmd", "resnet-bottleneck.mlmd", "unet.mlmd"];

    mlmd_builtin_plugins::register_all();

    for name in &examples {
        let source = read_example(name);
        let tokens = mlmd_core::parser::tokenizer::tokenize(&source);
        assert!(
            !tokens.is_empty(),
            "Tokenizer should produce tokens for {}",
            name
        );
    }
}

/// Test that the parser can parse all example files without errors.
#[test]
fn test_all_examples_parse() {
    let examples = ["attention.mlmd", "inception.mlmd", "lenet.mlmd", "resnet-bottleneck.mlmd", "unet.mlmd"];

    mlmd_builtin_plugins::register_all();

    for name in &examples {
        let source = read_example(name);
        let tokens = mlmd_core::parser::tokenizer::tokenize(&source);
        let parse_result = mlmd_core::parser::parser::parse(&tokens);
        assert!(
            parse_result.errors.is_empty(),
            "Parse errors for {}: {:?}",
            name,
            parse_result.errors
        );
    }
}

/// Test the full pipeline on the ResNet bottleneck example.
#[test]
fn test_resnet_bottleneck_full_pipeline() {
    let source = read_example("resnet-bottleneck.mlmd");

    mlmd_builtin_plugins::register_all();
    mlmd_core::codegen::targets::register_all_codegen_targets();

    let tokens = mlmd_core::parser::tokenizer::tokenize(&source);
    let parse_result = mlmd_core::parser::parser::parse(&tokens);
    assert!(
        parse_result.errors.is_empty(),
        "Parse errors: {:?}",
        parse_result.errors
    );

    let graph = mlmd_core::parser::graph_builder::build_graph(&parse_result.nodes);
    assert!(!graph.blocks.is_empty(), "Graph should have blocks");

    let registry = mlmd_core::block::registry::build_block_registry();
    let shape_result = mlmd_core::shape::infer_shapes(&graph, &registry);
    assert!(
        shape_result.errors.is_empty(),
        "Shape inference errors: {:?}",
        shape_result.errors
    );
}

/// Test the full pipeline on the UNet example.
#[test]
fn test_unet_full_pipeline() {
    let source = read_example("unet.mlmd");

    mlmd_builtin_plugins::register_all();
    mlmd_core::codegen::targets::register_all_codegen_targets();

    let tokens = mlmd_core::parser::tokenizer::tokenize(&source);
    let parse_result = mlmd_core::parser::parser::parse(&tokens);
    assert!(
        parse_result.errors.is_empty(),
        "Parse errors: {:?}",
        parse_result.errors
    );

    let graph = mlmd_core::parser::graph_builder::build_graph(&parse_result.nodes);
    assert!(!graph.blocks.is_empty(), "Graph should have blocks");

    let registry = mlmd_core::block::registry::build_block_registry();
    let shape_result = mlmd_core::shape::infer_shapes(&graph, &registry);
    assert!(
        shape_result.errors.is_empty(),
        "Shape inference errors: {:?}",
        shape_result.errors
    );
}
