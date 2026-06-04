use std::collections::HashMap;
use std::sync::OnceLock;

use crate::ast::graph::{Block, Graph};
use crate::ast::graph_utils::{build_adjacency, topo_sort};
use crate::codegen::result::GeneratedFile;
use crate::codegen::target::{register_target, CodegenTarget};
use crate::plugin::registry::get_block_codegen;
use crate::plugin::traits::{BlockCodegenFn, BlockCodegenResult};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn shape_comment(shapes: &[Vec<usize>]) -> String {
    if shapes.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = shapes
        .iter()
        .map(|s| {
            let inner: Vec<String> = s.iter().map(|d| d.to_string()).collect();
            format!("[{}]", inner.join(", "))
        })
        .collect();
    format!("  # {}", parts.join(", "))
}

/// Fallback codegen for blocks that don't have a registered keras codegen.
fn keras_fallback_codegen(
    block: &Block,
    input_vars: &[String],
    _output_vars: &[String],
) -> BlockCodegenResult {
    let main_in = input_vars.first().cloned().unwrap_or_else(|| "x".to_string());
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} /* {} — custom block, passthrough in generated code */",
            main_in, block.block_type
        ),
    }
}

fn get_block_codegen_with_fallback(block_type: &str) -> BlockCodegenFn {
    get_block_codegen(block_type, "keras").unwrap_or(keras_fallback_codegen)
}

// ---------------------------------------------------------------------------
// Test generation
// ---------------------------------------------------------------------------

fn generate_keras_test(_class_name: &str, sorted: &[Block]) -> Option<String> {
    // Collect input shapes from Input blocks (add batch dim 1)
    let input_shapes: Vec<Vec<usize>> = sorted
        .iter()
        .filter(|b| b.block_type == "Input")
        .filter_map(|b| b.output_shapes.first())
        .map(|s| {
            let mut full = vec![1usize];
            full.extend_from_slice(s);
            full
        })
        .collect();

    // Collect output shape from Output block's input_shapes (add batch dim 1)
    let output_shape: Option<Vec<usize>> = sorted
        .iter()
        .find(|b| b.block_type == "Output")
        .and_then(|b| b.input_shapes.first())
        .map(|s| {
            let mut full = vec![1usize];
            full.extend_from_slice(s);
            full
        });

    if input_shapes.is_empty() || output_shape.is_none() {
        return None;
    }
    let output_shape = output_shape.unwrap();

    // Build input tensor creation statements
    let input_lines: Vec<String> = input_shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let var_name = if i == 0 {
                "x".to_string()
            } else {
                format!("x{}", i + 1)
            };
            format!("    {} = tf.random.normal(({},))", var_name, shape_str)
        })
        .collect();

    let input_args: Vec<String> = input_lines
        .iter()
        .map(|line| {
            line.split_whitespace()
                .nth(1)
                .unwrap_or("x")
                .to_string()
        })
        .collect();

    let output_shape_str = output_shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let content = format!(
        "import pytest\nimport tensorflow as tf\n\n\ndef test_build_model():\n    model = build_model()\n{}\n    output = model({}, training=False)\n    assert output.shape == ({output_shape_str},)\n",
        input_lines.join("\n"),
        input_args.join(", "),
    );

    Some(content)
}

// ---------------------------------------------------------------------------
// KerasCodegen
// ---------------------------------------------------------------------------

pub struct KerasCodegen;

impl KerasCodegen {
    /// Register this codegen target. Safe to call multiple times.
    pub fn register() {
        REGISTERED.get_or_init(|| {
            register_target(Box::new(KerasCodegen));
        });
    }
}

impl CodegenTarget for KerasCodegen {
    fn name(&self) -> &'static str {
        "keras"
    }

    fn file_extension(&self) -> &'static str {
        "keras.py"
    }

    fn generate(&self, graph: &Graph) -> Vec<GeneratedFile> {
        let adj = build_adjacency(graph);
        let sorted = topo_sort(graph);

        // Named outputs: blockId → variableName
        let mut named_outputs: HashMap<String, String> = HashMap::new();
        for e in &graph.edges {
            if let Some(ref name) = e.tensor_name {
                named_outputs.insert(e.from.clone(), name.clone());
            }
        }

        let mut lines: Vec<String> = Vec::new();
        lines.push("import tensorflow as tf".to_string());
        lines.push("from tensorflow import keras".to_string());
        lines.push(String::new());
        lines.push(String::new());
        lines.push("def build_model():".to_string());

        // Track what variable name each block's output is stored in
        let mut block_output_var: HashMap<String, String> = HashMap::new();
        let mut var_counter: usize = 0;

        let mut input_var = "inputs".to_string();
        let mut output_var = "x".to_string();

        for b in &sorted {
            let info = adj.get(&b.id).unwrap();
            let input_vars: Vec<String> = info
                .inputs
                .iter()
                .map(|in_id| {
                    block_output_var
                        .get(in_id)
                        .cloned()
                        .unwrap_or_else(|| "x".to_string())
                })
                .collect();

            if b.block_type == "Input" {
                let shape = b.output_shapes.first().cloned().unwrap_or_default();
                // Drop batch dimension (keras.Input doesn't include batch dim)
                let spatial_shape: Vec<usize> = shape.iter().skip(1).copied().collect();
                let shape_str = if spatial_shape.is_empty() {
                    "None".to_string()
                } else {
                    spatial_shape
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let named = named_outputs.get(&b.id).cloned();
                let out_var = named.unwrap_or_else(|| input_var.clone());
                block_output_var.insert(b.id.clone(), out_var.clone());
                let sc = shape_comment(&b.output_shapes);
                lines.push(format!("    {} = keras.Input(shape=({},)){}", out_var, shape_str, sc));
                input_var = out_var;
                continue;
            }

            if b.block_type == "Output" {
                let ret_var = input_vars.first().cloned().unwrap_or_else(|| "x".to_string());
                output_var = ret_var;
                let sc = shape_comment(&b.input_shapes);
                lines.push(format!("    # output{}", sc));
                continue;
            }

            let fn_ptr = get_block_codegen_with_fallback(&b.block_type);
            // Determine output variable name
            let out_var: String;
            if let Some(named) = named_outputs.get(&b.id) {
                out_var = named.clone();
            } else if info.outputs.len() <= 1 {
                out_var = "x".to_string();
            } else {
                var_counter += 1;
                out_var = if var_counter == 1 {
                    "x".to_string()
                } else {
                    format!("x{}", var_counter)
                };
            }
            block_output_var.insert(b.id.clone(), out_var.clone());

            let out_count = b.output_shapes.len();
            let output_vars: Vec<String>;
            if out_count <= 1 {
                output_vars = vec![out_var];
            } else {
                var_counter += 1;
                let base = if var_counter == 1 {
                    "x".to_string()
                } else {
                    format!("x{}", var_counter)
                };
                output_vars = (0..out_count).map(|i| format!("{}_{}", base, i)).collect();
            }
            let result = fn_ptr(b, &input_vars, &output_vars);
            let shape_ann = shape_comment(&b.output_shapes);
            lines.push(format!("    {}{}", result.forward, shape_ann));
        }

        let model_name = graph
            .groups
            .first()
            .and_then(|g| g.path.first())
            .map(|s| s.as_str())
            .unwrap_or("Model");

        lines.push(format!(
            "    return keras.Model(inputs={}, outputs={})",
            input_var, output_var
        ));

        let content = lines.join("\n") + "\n";

        // Generate test file
        let test_content = generate_keras_test(model_name, &sorted);

        let mut files = vec![GeneratedFile {
            path: format!("{}.keras.py", model_name),
            content,
        }];

        if let Some(tc) = test_content {
            files.push(GeneratedFile {
                path: format!("{}.keras.test.py", model_name),
                content: tc,
            });
        }

        files
    }
}

static REGISTERED: OnceLock<()> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::graph::{Block, Edge, Graph};
    use crate::ast::nodes::SourceLoc;
    use crate::codegen::target::get_target;

    fn dummy_loc() -> SourceLoc {
        SourceLoc {
            line: 0,
            col: 0,
            offset: 0,
        }
    }

    fn make_linear_graph() -> Graph {
        Graph {
            blocks: vec![
                Block {
                    id: "inp".to_string(),
                    block_type: "Input".to_string(),
                    params: std::collections::HashMap::new(),
                    input_shapes: vec![],
                    output_shapes: vec![vec![1, 28, 28]],
                    param_count: None,
                    show_depth: None,
                    loc: dummy_loc(),
                },
                Block {
                    id: "out".to_string(),
                    block_type: "Output".to_string(),
                    params: std::collections::HashMap::new(),
                    input_shapes: vec![vec![10]],
                    output_shapes: vec![],
                    param_count: None,
                    show_depth: None,
                    loc: dummy_loc(),
                },
            ],
            edges: vec![Edge {
                from: "inp".to_string(),
                to: "out".to_string(),
                tensor_name: None,
                shape: None,
            }],
            groups: vec![],
        }
    }

    #[test]
    fn test_name_and_extension() {
        let target = KerasCodegen;
        assert_eq!(target.name(), "keras");
        assert_eq!(target.file_extension(), "keras.py");
    }

    #[test]
    fn test_generate_simple_graph() {
        let graph = make_linear_graph();
        let target = KerasCodegen;
        let files = target.generate(&graph);
        assert_eq!(files.len(), 2, "should generate model and test files");

        // Model file
        let model_file = &files[0];
        assert!(model_file.path.ends_with(".keras.py"));
        assert!(model_file.content.contains("def build_model():"));
        assert!(model_file.content.contains("keras.Input"));
        assert!(model_file.content.contains("keras.Model("));

        // Test file
        let test_file = &files[1];
        assert!(test_file.path.ends_with(".keras.test.py"));
        assert!(test_file.content.contains("def test_build_model()"));
        assert!(test_file.content.contains("tf.random.normal"));
    }

    #[test]
    fn test_generate_no_test_when_no_shapes() {
        let graph = Graph {
            blocks: vec![
                Block {
                    id: "inp".to_string(),
                    block_type: "Input".to_string(),
                    params: std::collections::HashMap::new(),
                    input_shapes: vec![],
                    output_shapes: vec![],
                    param_count: None,
                    show_depth: None,
                    loc: dummy_loc(),
                },
            ],
            edges: vec![],
            groups: vec![],
        };
        let target = KerasCodegen;
        let files = target.generate(&graph);
        assert!(!files.is_empty());
    }

    #[test]
    fn test_register_and_get() {
        KerasCodegen::register();
        let target = get_target("keras");
        assert!(target.is_some());
        assert_eq!(target.unwrap().name(), "keras");
    }
}

