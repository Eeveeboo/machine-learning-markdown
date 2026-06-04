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
    assert!(
        !parse_result.nodes.is_empty(),
        "Parser should produce nodes"
    );

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
    let examples = [
        "attention.mlmd",
        "inception.mlmd",
        "lenet.mlmd",
        "resnet-bottleneck.mlmd",
        "scale.mlmd",
        "unet.mlmd",
    ];

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
    let examples = [
        "attention.mlmd",
        "inception.mlmd",
        "lenet.mlmd",
        "resnet-bottleneck.mlmd",
        "scale.mlmd",
        "unet.mlmd",
    ];

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

// =========================================================================
// Example plugin (Scale) integration tests
//
// These tests validate:
//   1. Static registration: a custom block works with the full pipeline
//   2. Dynamic loading: the cdylib FFI path works end-to-end
// =========================================================================

/// Full pipeline test using the Scale example plugin registered statically.
///
/// This validates that a user-defined BlockDef + codegen functions integrate
/// correctly with shape inference and all three codegen targets.
#[test]
fn test_scale_static_pipeline() {
    let source = read_example("scale.mlmd");

    // Register builtins + codegen targets + example plugin
    mlmd_builtin_plugins::register_all();
    mlmd_core::codegen::targets::register_all_codegen_targets();
    mlmd_example_plugin::register();

    // 1. Tokenize
    let tokens = mlmd_core::parser::tokenizer::tokenize(&source);
    assert!(!tokens.is_empty(), "Tokenizer should produce tokens");

    // 2. Parse
    let parse_result = mlmd_core::parser::parser::parse(&tokens);
    assert!(
        parse_result.errors.is_empty(),
        "Parse errors: {:?}",
        parse_result.errors
    );

    // 3. Build graph
    let graph = mlmd_core::parser::graph_builder::build_graph(&parse_result.nodes);
    assert!(!graph.blocks.is_empty(), "Graph should have blocks");

    // Verify the graph contains a Scale block
    let scale_blocks: Vec<_> = graph
        .blocks
        .iter()
        .filter(|b| b.block_type == "Scale")
        .collect();
    assert_eq!(scale_blocks.len(), 1, "Graph should have one Scale block");
    assert_eq!(
        scale_blocks[0].params.get("factor").and_then(|v| {
            if let mlmd_core::ast::nodes::ParamValue::Number(n) = v {
                Some(n.value)
            } else {
                None
            }
        }),
        Some(2.0),
        "Scale block should have factor=2.0"
    );

    // 4. Shape inference
    let registry = mlmd_core::block::registry::build_block_registry();
    let shape_result = mlmd_core::shape::infer_shapes(&graph, &registry);
    assert!(
        shape_result.errors.is_empty(),
        "Shape inference errors: {:?}",
        shape_result.errors
    );

    // Scale should pass through the shape unchanged
    let scale_b = shape_result
        .graph
        .blocks
        .iter()
        .find(|b| b.block_type == "Scale")
        .expect("Scale block should exist in shaped graph");
    assert_eq!(
        scale_b.input_shapes,
        vec![vec![1, 3, 32, 32]],
        "Scale input should be [1,3,32,32]"
    );
    assert_eq!(
        scale_b.output_shapes,
        vec![vec![1, 3, 32, 32]],
        "Scale output should match input (passthrough)"
    );
    assert_eq!(
        scale_b.param_count,
        Some(1),
        "Scale should report 1 parameter"
    );

    // 5. Codegen produces output for all targets
    for target_name in &["pytorch", "candle", "keras"] {
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

        // Verify the Scale block appears in the generated code
        let all_content: String = files.iter().map(|f| f.content.as_str()).collect();
        assert!(
            all_content.contains("Scale") || all_content.contains("scale"),
            "Generated code for '{}' should contain Scale reference",
            target_name
        );
    }
}

/// Test dynamic plugin loading via the FFI/cdylib path.
///
/// This validates that a .dylib/.so built from the example plugin crate can
/// be loaded at runtime, registered via the adapter, and used transparently
/// in the MLMD pipeline.
///
/// Note: The cdylib must be built first:
///   cargo build -p mlmd-example-plugin
#[test]
fn test_dynamic_plugin_loading() {
    // Find the built cdylib
    let plugin_path = find_example_plugin_cdylib();
    if plugin_path.is_none() {
        eprintln!(
            "Skipping dynamic plugin test: cdylib not found. \
             Build with: cargo build -p mlmd-example-plugin"
        );
        return;
    }
    let plugin_path = plugin_path.unwrap();

    // Register builtins + codegen targets
    mlmd_builtin_plugins::register_all();
    mlmd_core::codegen::targets::register_all_codegen_targets();

    // Load the dynamic plugin
    let block_name =
        mlmd_core::plugin::adapter::register_dynamic_plugin(plugin_path.to_str().unwrap())
            .expect("Failed to register dynamic plugin");
    assert_eq!(
        block_name, "Scale",
        "Dynamic plugin should advertise 'Scale'"
    );

    // Verify the block is registered in the BlockDef registry
    let def = mlmd_core::block::registry::lookup_block("Scale");
    assert!(
        def.is_some(),
        "Scale should be registered via dynamic plugin"
    );
    assert_eq!(def.unwrap().name(), "Scale");

    // Verify codegen functions are registered
    for target in &["pytorch", "candle", "keras"] {
        let codegen = mlmd_core::plugin::registry::get_block_codegen("Scale", target);
        assert!(
            codegen.is_some(),
            "Codegen for Scale/{target} should be registered"
        );
    }

    // Full pipeline with scale.mlmd
    let source = read_example("scale.mlmd");
    let tokens = mlmd_core::parser::tokenizer::tokenize(&source);
    let parse_result = mlmd_core::parser::parser::parse(&tokens);
    assert!(
        parse_result.errors.is_empty(),
        "Parse errors: {:?}",
        parse_result.errors
    );

    let graph = mlmd_core::parser::graph_builder::build_graph(&parse_result.nodes);
    let registry = mlmd_core::block::registry::build_block_registry();
    let shape_result = mlmd_core::shape::infer_shapes(&graph, &registry);
    assert!(
        shape_result.errors.is_empty(),
        "Shape inference errors: {:?}",
        shape_result.errors
    );

    // Verify Scale output shape
    let scale_b = shape_result
        .graph
        .blocks
        .iter()
        .find(|b| b.block_type == "Scale")
        .expect("Scale block should exist");
    assert_eq!(scale_b.output_shapes, vec![vec![1, 3, 32, 32]]);

    // Verify codegen works through FFI
    for target_name in &["pytorch", "candle", "keras"] {
        let result = mlmd_core::codegen::target::with_target(target_name, |target| {
            target.generate(&shape_result.graph)
        });
        assert!(
            result.is_some(),
            "Codegen target '{target_name}' should exist"
        );
        let files = result.unwrap();
        assert!(
            !files.is_empty(),
            "Codegen target '{target_name}' should produce files"
        );
    }
}

/// Locate the built `libmlmd_example_plugin` cdylib.
///
/// Checks `target/debug/` and `target/release/` relative to the workspace root.
/// Returns `None` if the library hasn't been built yet.
fn find_example_plugin_cdylib() -> Option<PathBuf> {
    let root = workspace_root();

    let lib_name = format!("libmlmd_example_plugin{}", std::env::consts::DLL_SUFFIX);

    // Check debug first, then release
    for profile in &["debug", "release"] {
        let path = root.join("target").join(profile).join(&lib_name);
        if path.exists() {
            return Some(path);
        }
    }

    None
}
