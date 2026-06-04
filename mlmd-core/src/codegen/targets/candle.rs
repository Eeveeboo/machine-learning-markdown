use std::collections::HashMap;
use std::sync::OnceLock;

use crate::ast::graph::{Block, Graph};
use crate::ast::graph_utils::{build_adjacency, topo_sort};
use crate::codegen::result::GeneratedFile;
use crate::codegen::target::{register_target, CodegenTarget};
use crate::plugin::registry::get_block_codegen;
use crate::plugin::traits::CandleInitOrString;
use crate::plugin::traits::{BlockCodegenFn, BlockCodegenResult};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn indent_lines(s: &str, indent: &str) -> String {
    s.lines()
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("{}{}", indent, line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Convert first character to lowercase (matches TS toSnakeCase).
fn to_snake_case(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_lowercase().to_string() + chars.as_str(),
    }
}

/// Fallback codegen for blocks that don't have a registered candle codegen.
fn candle_fallback_codegen(
    block: &Block,
    input_vars: &[String],
    _output_vars: &[String],
) -> BlockCodegenResult {
    let main_in = input_vars
        .first()
        .cloned()
        .unwrap_or_else(|| "x".to_string());
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} /* {} — custom block, passthrough in generated code */",
            main_in, block.block_type
        ),
    }
}

fn get_block_codegen_with_fallback(block_type: &str) -> BlockCodegenFn {
    get_block_codegen(block_type, "candle").unwrap_or(candle_fallback_codegen)
}

// ---------------------------------------------------------------------------
// Test module generation
// ---------------------------------------------------------------------------

/// Generate a `#[cfg(test)] mod tests { ... }` block for Rust Candle models.
/// Contains `test_forward` (validates forward pass output shape) and
/// `test_save_load` (validates weight save/load round-trip).
fn generate_test_module(
    class_name: &str,
    forward_params: &[String],
    input_shapes: &[Vec<usize>],
    output_shape: Option<&[usize]>,
) -> String {
    // Edge cases: skip if no output shape or no forward params
    let output_shape = match output_shape {
        Some(s) => s,
        None => return String::new(),
    };
    if forward_params.is_empty() {
        return String::new();
    }

    // Extract param variable names: "x: &Tensor" → "x"
    let param_names: Vec<&str> = forward_params
        .iter()
        .map(|p| p.split(':').next().unwrap().trim())
        .collect();

    // Determine which shape to use for each param.
    // Defensive: if lengths mismatch, use input_shapes[0] for all params.
    let shapes_for_params: Vec<&[usize]> = if forward_params.len() != input_shapes.len() {
        let default = input_shapes.first().map(|s| s.as_slice()).unwrap_or(&[]);
        (0..forward_params.len()).map(|_| default).collect()
    } else {
        input_shapes.iter().map(|s| s.as_slice()).collect()
    };

    // Helper: format &[a, b, c]
    let format_shape = |shape: &[usize]| -> String {
        if shape.is_empty() {
            "&[]".to_string()
        } else {
            let inner: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("&[{}]", inner.join(", "))
        }
    };

    // Build `let param = Tensor::randn(0f32, 1.0, &[shape], &dev)?;` lines
    let input_tensor_lines: Vec<String> = param_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let shape_slice = format_shape(shapes_for_params[i]);
            format!(
                "        let {} = candle_core::Tensor::randn(0f32, 1.0, {}, &dev)?;",
                name, shape_slice
            )
        })
        .collect();
    let input_tensors = input_tensor_lines.join("\n");

    // Build forward call arguments: &x, &query, &key, ...
    let forward_args: Vec<String> = param_names
        .iter()
        .map(|name| format!("&{}", name))
        .collect();
    let forward_args_str = forward_args.join(", ");

    // Build output shape slice
    let output_shape_slice = format_shape(output_shape);

    format!(
        r#"

#[allow(warnings)]
#[cfg(test)]
mod tests {{
    use super::*;
    use candle_nn::VarMap;

    fn setup() -> (candle_core::Device, VarMap, candle_nn::VarBuilder<'static>) {{
        let dev = candle_core::Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        (dev, varmap, vb)
    }}

    #[test]
    fn test_forward() -> candle_core::Result<()> {{
        let (dev, _varmap, vb) = setup();
        let model = {class_name}::new(vb)?;
{input_tensors}
        let output = model.forward({forward_args_str})?;
        assert_eq!(output.dims(), {output_shape_slice});
        Ok(())
    }}

    #[test]
    fn test_save_load() -> candle_core::Result<()> {{
        let (dev, varmap, vb) = setup();
        let model = {class_name}::new(vb)?;
{input_tensors}
        let output_before = model.forward({forward_args_str})?;

        let dir = std::env::temp_dir().join("mlmd-e2e");
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let path = dir.join("{class_name}.safetensors");
        let _ = std::fs::remove_file(&path);
        varmap.save(&path)?;

        let mut varmap2 = VarMap::new();
        let vb2 = candle_nn::VarBuilder::from_varmap(&varmap2, candle_core::DType::F32, &dev);
        let model2 = {class_name}::new(vb2)?;
        varmap2.load(&path)?;
        let output_after = model2.forward({forward_args_str})?;

        let diff = (output_before - &output_after)?.abs()?.sum_all()?;
        assert!(diff.to_vec0::<f32>()? < 1e-5);

        let _ = std::fs::remove_file(&path);
        Ok(())
    }}
}}
"#,
        class_name = class_name,
        input_tensors = input_tensors,
        forward_args_str = forward_args_str,
        output_shape_slice = output_shape_slice,
    )
}

// ---------------------------------------------------------------------------
// CandleCodegen
// ---------------------------------------------------------------------------

pub struct CandleCodegen;

impl CandleCodegen {
    /// Register this codegen target. Safe to call multiple times.
    pub fn register() {
        REGISTERED.get_or_init(|| {
            register_target(Box::new(CandleCodegen));
        });
    }
}

impl CodegenTarget for CandleCodegen {
    fn name(&self) -> &'static str {
        "candle"
    }

    fn file_extension(&self) -> &'static str {
        "_candle.rs"
    }

    fn generate(&self, graph: &Graph) -> Vec<GeneratedFile> {
        let adj = build_adjacency(graph);
        let sorted = topo_sort(graph);

        let named_outputs: HashMap<String, String> = graph
            .edges
            .iter()
            .filter_map(|e| e.tensor_name.as_ref().map(|n| (e.from.clone(), n.clone())))
            .collect();

        // Collect struct fields (learnable layers)
        let mut struct_field_lines: Vec<String> = Vec::new();
        let mut with_scopes_init_lines: Vec<String> = Vec::new();
        let mut new_ok_fields: Vec<String> = Vec::new();
        let mut default_scope_args: Vec<String> = Vec::new();
        let mut scope_params: Vec<String> = Vec::new();

        for b in &sorted {
            if b.block_type == "Input" || b.block_type == "Output" {
                continue;
            }
            let fn_ptr = get_block_codegen_with_fallback(&b.block_type);
            let result = fn_ptr(b, &[], &[]); // init pass: no var names needed

            if let Some(ref init) = result.init {
                if let CandleInitOrString::Candle(ref candle_init) = init {
                    struct_field_lines.push(format!("    {},", candle_init.field));

                    // Transform vb.pp("blockId") → weights.pp(snakeName) for with_scopes
                    let snake_name = to_snake_case(&b.id);
                    let body = candle_init.body.replace(
                        &format!("vb.pp(\"{}\")", b.id),
                        &format!("weights.pp({})", snake_name),
                    );
                    with_scopes_init_lines.push(indent_lines(&body, "        "));

                    new_ok_fields.push(format!("            {},", b.id));
                    default_scope_args.push(format!("        \"{}\",", b.id));

                    // Derive scope param name from field name
                    let field_name = candle_init
                        .field
                        .split(':')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let scope_param_name = to_snake_case(&field_name);
                    scope_params.push(format!("        {}: &str,", scope_param_name));
                }
            }
        }

        // Collect Input blocks → forward parameter names
        let mut input_params: HashMap<String, String> = HashMap::new();
        let mut forward_params: Vec<String> = Vec::new();
        let mut unnamed_count: usize = 0;

        for b in &sorted {
            if b.block_type != "Input" {
                continue;
            }
            if let Some(named) = named_outputs.get(&b.id) {
                input_params.insert(b.id.clone(), named.clone());
                forward_params.push(format!("{}: &Tensor", named));
            } else {
                unnamed_count += 1;
                let name = if unnamed_count == 1 {
                    "x".to_string()
                } else {
                    format!("x{}", unnamed_count)
                };
                input_params.insert(b.id.clone(), name.clone());
                forward_params.push(format!("{}: &Tensor", name));
            }
        }

        // Track output variable per block
        let mut block_output_var: HashMap<String, String> = HashMap::new();
        let mut var_counter: usize = 0;

        let mut forward_lines: Vec<String> = Vec::new();

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
                block_output_var.insert(
                    b.id.clone(),
                    input_params
                        .get(&b.id)
                        .cloned()
                        .unwrap_or_else(|| "x".to_string()),
                );
                continue;
            }

            if b.block_type == "Output" {
                let ret_var = input_vars
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "x".to_string());
                forward_lines.push(format!("        Ok({})", ret_var));
                continue;
            }

            let fn_ptr = get_block_codegen_with_fallback(&b.block_type);
            let out_count = b.output_shapes.len();
            let output_vars: Vec<String>;

            if out_count <= 1 {
                if let Some(named) = named_outputs.get(&b.id) {
                    output_vars = vec![named.clone()];
                } else if info.outputs.len() == 1 {
                    output_vars = vec!["x".to_string()];
                } else {
                    var_counter += 1;
                    let name = if var_counter == 1 {
                        "x".to_string()
                    } else {
                        format!("x{}", var_counter)
                    };
                    output_vars = vec![name];
                }
            } else {
                var_counter += 1;
                let base = if var_counter == 1 {
                    "x".to_string()
                } else {
                    format!("x{}", var_counter)
                };
                output_vars = (0..out_count).map(|i| format!("{}_{}", base, i)).collect();
            }
            block_output_var.insert(b.id.clone(), output_vars[0].clone());

            let result = fn_ptr(b, &input_vars, &output_vars);
            // Wrap in `let` since plugin templates provide `{var} = expr?;` but not the `let` keyword
            let line = if result.forward.starts_with("let ") {
                result.forward.clone()
            } else {
                format!("let {}", result.forward)
            };
            forward_lines.push(indent_lines(&line, "        "));
        }

        let class_name = graph
            .groups
            .first()
            .and_then(|g| g.path.first())
            .map(|s| s.as_str())
            .unwrap_or("Model");

        // Forward signature
        let forward_param_block = if forward_params.is_empty() {
            "        // no inputs defined".to_string()
        } else {
            format!("        {}", forward_params.join(",\n        "))
        };

        let mut lines: Vec<String> = Vec::new();
        lines.push("// This file was generated by mlmd. Do not edit.".to_string());
        lines.push(String::new());
        lines.push("#[allow(warnings)]".to_string());
        lines.push("use candle_core::{ModuleT, Result, Tensor};".to_string());
        lines.push("use candle_nn::{Module, VarBuilder};".to_string());
        lines.push(String::new());
        lines.push("#[allow(warnings)]".to_string());
        lines.push(format!("pub struct {} {{", class_name));
        for l in &struct_field_lines {
            lines.push(l.clone());
        }
        lines.push("}".to_string());
        lines.push(String::new());
        lines.push("#[allow(warnings)]".to_string());
        lines.push(format!("impl {} {{", class_name));
        lines.push(format!(
            "    /// Create model with auto-generated weight scope names."
        ));
        lines.push(format!(
            "    /// Use [{}::with_scopes] for custom scope names.",
            class_name
        ));
        lines.push(format!(
            "    pub fn new(weights: VarBuilder) -> Result<Self> {{"
        ));
        lines.push(format!("        Self::with_scopes(weights,"));
        for l in &default_scope_args {
            lines.push(l.clone());
        }
        lines.push("        )".to_string());
        lines.push("    }".to_string());
        lines.push(String::new());
        lines.push(format!(
            "    /// Create model with custom weight-loading scope names."
        ));
        lines.push(format!("    pub fn with_scopes("));
        lines.push("        weights: VarBuilder,".to_string());
        for l in &scope_params {
            lines.push(l.clone());
        }
        lines.push("    ) -> Result<Self> {".to_string());
        for l in &with_scopes_init_lines {
            lines.push(l.clone());
        }
        lines.push("        Ok(Self {".to_string());
        for l in &new_ok_fields {
            lines.push(l.clone());
        }
        lines.push("        })".to_string());
        lines.push("    }".to_string());
        lines.push(String::new());
        lines.push(format!("    pub fn forward(&self,"));
        lines.push(forward_param_block.clone());
        lines.push("    ) -> Result<Tensor> {".to_string());
        for l in &forward_lines {
            lines.push(l.clone());
        }
        lines.push("    }".to_string());
        lines.push("}".to_string());

        let content = lines.join("\n") + "\n";

        // Build test module
        // Prepend batch dimension (1) to all shapes since MLMD shapes don't include batch dim
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

        // Get output shape from the Output block's input_shapes
        let output_block = sorted.iter().find(|b| b.block_type == "Output");
        let output_shape_base = output_block.and_then(|b| b.input_shapes.first());
        let output_shape: Option<Vec<usize>> = output_shape_base.map(|s| {
            let mut full = vec![1usize];
            full.extend_from_slice(s);
            full
        });

        let test_module = generate_test_module(
            class_name,
            &forward_params,
            &input_shapes,
            output_shape.as_deref(),
        );
        let final_content = if test_module.is_empty() {
            content.clone()
        } else {
            content + &test_module
        };

        vec![GeneratedFile {
            path: format!("{}_candle.rs", class_name),
            content: final_content,
        }]
    }
}

static REGISTERED: OnceLock<()> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::graph::{Block, Edge, Graph, Group};
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
        let target = CandleCodegen;
        assert_eq!(target.name(), "candle");
        assert_eq!(target.file_extension(), "_candle.rs");
    }

    #[test]
    fn test_generate_simple_graph() {
        let graph = make_linear_graph();
        let target = CandleCodegen;
        let files = target.generate(&graph);
        assert_eq!(files.len(), 1, "should generate one file");

        let file = &files[0];
        assert!(file.path.ends_with("_candle.rs"));
        assert!(file
            .content
            .starts_with("// This file was generated by mlmd. Do not edit."));
        assert!(file.content.contains("pub struct Model {"));
        assert!(file.content.contains("impl Model {"));
        assert!(file.content.contains("pub fn new(weights: VarBuilder)"));
        assert!(file.content.contains("pub fn with_scopes("));
        assert!(file.content.contains("pub fn forward(&self,"));
    }

    #[test]
    fn test_generate_with_group_name() {
        let mut graph = make_linear_graph();
        graph.groups = vec![Group {
            path: vec!["MyModel".to_string()],
            block_ids: vec![],
        }];
        let target = CandleCodegen;
        let files = target.generate(&graph);
        assert!(files[0].content.contains("pub struct MyModel {"));
    }

    #[test]
    fn test_generate_no_test_when_no_output_shape() {
        let graph = Graph {
            blocks: vec![Block {
                id: "inp".to_string(),
                block_type: "Input".to_string(),
                params: std::collections::HashMap::new(),
                input_shapes: vec![],
                output_shapes: vec![vec![1, 28, 28]],
                param_count: None,
                show_depth: None,
                loc: dummy_loc(),
            }],
            edges: vec![],
            groups: vec![],
        };
        let target = CandleCodegen;
        let files = target.generate(&graph);
        // Should generate model without test module
        assert_eq!(files.len(), 1);
        assert!(!files[0].content.contains("#[cfg(test)]"));
    }

    #[test]
    fn test_register_and_get() {
        CandleCodegen::register();
        let target = get_target("candle");
        assert!(target.is_some());
        assert_eq!(target.unwrap().name(), "candle");
    }

    #[test]
    fn test_generate_test_module_content() {
        let graph = make_linear_graph();
        let target = CandleCodegen;
        let files = target.generate(&graph);
        let content = &files[0].content;
        // With output shape [10], test module should be generated
        assert!(content.contains("#[cfg(test)]"));
        assert!(content.contains("fn test_forward()"));
        assert!(content.contains("fn test_save_load()"));
        assert!(content.contains("Tensor::randn"));
        assert!(content.contains("assert_eq!(output.dims(), &[1, 10])"));
    }
}
