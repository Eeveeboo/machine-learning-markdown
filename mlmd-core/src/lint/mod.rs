pub mod rules;

use crate::ast::graph::Graph;
use crate::ast::nodes::SourceLoc;
use crate::lint::rules::*;
use crate::shape::BlockRegistry;

// ---------------------------------------------------------------------------
// LintDiagnostic — a single diagnostic produced by linting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct LintDiagnostic {
    pub severity: Severity,
    pub message: String,
    pub loc: SourceLoc,
    pub rule: String,
}

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    Error,
    Warning,
}

// ---------------------------------------------------------------------------
// lint — run all 7 lint rules against the graph
// ---------------------------------------------------------------------------

/// Run all lint rules against the graph.
/// Cycle detection runs first; shape checks are skipped if cycles are present.
pub fn lint(graph: &Graph, registry: &BlockRegistry) -> Vec<LintDiagnostic> {
    let mut diags: Vec<LintDiagnostic> = Vec::new();

    // 1. Cycle detection must run first
    let cycle_errors = check_cycles(graph);
    let has_cycles = !cycle_errors.is_empty();
    diags.extend(cycle_errors);

    if has_cycles {
        // Skip shape inference and other order-dependent checks if cycles exist
        diags.extend(check_missing_plugins(graph, registry));
        diags.extend(check_missing_params(graph, registry));
        diags.extend(check_duplicate_tensor_names(graph));
        diags.extend(check_undefined_tensor_refs(graph));
        diags.extend(check_unused_tensors(graph));
        return diags;
    }

    // 2. Missing plugins
    diags.extend(check_missing_plugins(graph, registry));

    // 3. Missing required params
    diags.extend(check_missing_params(graph, registry));

    // 4. Shape compatibility (runs infer_shapes internally)
    diags.extend(check_shapes(graph, registry));

    // 5. Duplicate tensor names
    diags.extend(check_duplicate_tensor_names(graph));

    // 6. Undefined tensor references
    diags.extend(check_undefined_tensor_refs(graph));

    // 7. Unused named tensors
    diags.extend(check_unused_tensors(graph));

    diags
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::graph::{Block, Edge};
    use crate::ast::nodes::{BarewordVal, ListVal, NumberVal, ParamValue, ShapeVal, SourceLoc};
    use crate::block::types::{BlockDef, ParamSpec, ParamType};
    use std::collections::HashMap;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn dummy_loc() -> SourceLoc {
        SourceLoc {
            line: 0,
            col: 0,
            offset: 0,
        }
    }

    fn make_block(id: &str, block_type: &str) -> Block {
        Block {
            id: id.to_string(),
            block_type: block_type.to_string(),
            params: HashMap::new(),
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            param_count: None,
            show_depth: None,
            loc: dummy_loc(),
        }
    }

    fn make_block_with_params(
        id: &str,
        block_type: &str,
        params: HashMap<String, ParamValue>,
    ) -> Block {
        Block {
            id: id.to_string(),
            block_type: block_type.to_string(),
            params,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            param_count: None,
            show_depth: None,
            loc: dummy_loc(),
        }
    }

    fn make_edge(from: &str, to: &str) -> Edge {
        Edge {
            from: from.to_string(),
            to: to.to_string(),
            tensor_name: None,
            shape: None,
        }
    }

    fn make_edge_with_name(from: &str, to: &str, name: &str) -> Edge {
        Edge {
            from: from.to_string(),
            to: to.to_string(),
            tensor_name: Some(name.to_string()),
            shape: None,
        }
    }

    fn p_num(value: f64) -> ParamValue {
        ParamValue::Number(Box::new(NumberVal {
            kind: "number".to_string(),
            value,
            loc: dummy_loc(),
        }))
    }

    fn p_shape(dims: Vec<usize>) -> ParamValue {
        ParamValue::Shape(Box::new(ShapeVal {
            kind: "shape".to_string(),
            dims,
            loc: dummy_loc(),
        }))
    }

    fn p_bareword(s: &str) -> ParamValue {
        ParamValue::Bareword(Box::new(BarewordVal {
            kind: "bareword".to_string(),
            value: s.to_string(),
            loc: dummy_loc(),
        }))
    }

    fn p_list(items: Vec<ParamValue>) -> ParamValue {
        ParamValue::List(Box::new(ListVal {
            kind: "list".to_string(),
            items,
            loc: dummy_loc(),
        }))
    }

    fn empty_graph() -> Graph {
        Graph {
            blocks: vec![],
            edges: vec![],
            groups: vec![],
        }
    }

    // ------------------------------------------------------------------
    // Mock BlockDef implementations for testing
    // ------------------------------------------------------------------

    /// Input block: output shape from `dims` param
    struct MockInput;

    impl BlockDef for MockInput {
        fn name(&self) -> &str {
            "Input"
        }

        fn params(&self) -> &[ParamSpec] {
            &[]
        }

        fn infer_shape(
            &self,
            _inputs: &[crate::ast::graph::Shape],
            params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<crate::ast::graph::Shape>, String> {
            let dims = params
                .get("dims")
                .and_then(|v| {
                    if let ParamValue::Shape(s) = v {
                        Some(s.dims.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            Ok(vec![dims])
        }
    }

    /// ReLU: passthrough
    struct MockReLU;

    impl BlockDef for MockReLU {
        fn name(&self) -> &str {
            "ReLU"
        }

        fn params(&self) -> &[ParamSpec] {
            &[]
        }

        fn infer_shape(
            &self,
            inputs: &[crate::ast::graph::Shape],
            _params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<crate::ast::graph::Shape>, String> {
            Ok(inputs.to_vec())
        }
    }

    /// Add: passthrough, returns inputs[0]
    struct MockAdd;

    impl BlockDef for MockAdd {
        fn name(&self) -> &str {
            "Add"
        }

        fn params(&self) -> &[ParamSpec] {
            &[]
        }

        fn infer_shape(
            &self,
            inputs: &[crate::ast::graph::Shape],
            _params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<crate::ast::graph::Shape>, String> {
            if inputs.is_empty() {
                return Err("Add requires inputs".to_string());
            }
            Ok(vec![inputs[0].clone()])
        }
    }

    /// Conv2d: requires `kernel` param (and should have out_channels too for shape inference)
    struct MockConv2d;

    impl BlockDef for MockConv2d {
        fn name(&self) -> &str {
            "Conv2d"
        }

        fn params(&self) -> &[ParamSpec] {
            static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
                vec![
                    ParamSpec {
                        name: "kernel".into(),
                        param_type: ParamType::Number,
                        required: true,
                        default: None,
                    },
                    ParamSpec {
                        name: "out_channels".into(),
                        param_type: ParamType::Number,
                        required: true,
                        default: None,
                    },
                ]
            });
            &PARAMS
        }

        fn infer_shape(
            &self,
            _inputs: &[crate::ast::graph::Shape],
            params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<crate::ast::graph::Shape>, String> {
            let _kernel = params
                .get("kernel")
                .and_then(|v| {
                    if let ParamValue::Number(n) = v {
                        Some(n.value as usize)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| "Conv2d missing kernel".to_string())?;
            let _out_ch = params
                .get("out_channels")
                .and_then(|v| {
                    if let ParamValue::Number(n) = v {
                        Some(n.value as usize)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| "Conv2d missing out_channels".to_string())?;
            Ok(vec![vec![]])
        }
    }

    fn make_registry() -> BlockRegistry {
        let mut reg = BlockRegistry::new();
        reg.register(Box::new(MockInput));
        reg.register(Box::new(MockReLU));
        reg.register(Box::new(MockAdd));
        reg.register(Box::new(MockConv2d));
        reg
    }

    // ==============================================================
    // Tests
    // ==============================================================

    #[test]
    fn test_clean_graph() {
        let registry = make_registry();
        let mut params = HashMap::new();
        params.insert("dims".to_string(), p_shape(vec![3, 224, 224]));
        let graph = Graph {
            blocks: vec![
                make_block_with_params("b0", "Input", params),
                make_block("b1", "ReLU"),
            ],
            edges: vec![make_edge("b0", "b1")],
            groups: vec![],
        };
        let diags = lint(&graph, &registry);
        assert_eq!(diags.len(), 0, "expected no diagnostics, got: {:?}", diags);
    }

    #[test]
    fn test_clean_empty_graph() {
        let registry = make_registry();
        let diags = lint(&empty_graph(), &registry);
        assert_eq!(diags.len(), 0);
    }

    mod cycles {
        use super::*;

        #[test]
        fn test_simple_cycle() {
            let registry = make_registry();
            let graph = Graph {
                blocks: vec![make_block("b0", "ReLU"), make_block("b1", "ReLU")],
                edges: vec![make_edge("b0", "b1"), make_edge("b1", "b0")],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let cycle_diags: Vec<_> = diags.iter().filter(|d| d.rule == "cycle").collect();
            assert!(!cycle_diags.is_empty(), "expected cycle diagnostics");
            assert_eq!(cycle_diags[0].severity, Severity::Error);
        }

        #[test]
        fn test_self_loop() {
            let registry = make_registry();
            let graph = Graph {
                blocks: vec![make_block("b0", "ReLU")],
                edges: vec![make_edge("b0", "b0")],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let cycle_diags: Vec<_> = diags.iter().filter(|d| d.rule == "cycle").collect();
            assert!(!cycle_diags.is_empty(), "expected cycle diagnostics");
        }

        #[test]
        fn test_three_cycle() {
            let registry = make_registry();
            // a -> b -> c -> a
            let graph = Graph {
                blocks: vec![
                    make_block("a", "ReLU"),
                    make_block("b", "ReLU"),
                    make_block("c", "ReLU"),
                ],
                edges: vec![
                    make_edge("a", "b"),
                    make_edge("b", "c"),
                    make_edge("c", "a"),
                ],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let cycle_diags: Vec<_> = diags.iter().filter(|d| d.rule == "cycle").collect();
            assert_eq!(cycle_diags.len(), 3, "all 3 blocks should be in cycle");
            for d in &cycle_diags {
                assert_eq!(d.severity, Severity::Error);
            }
        }
    }

    mod missing_plugin {
        use super::*;

        #[test]
        fn test_unknown_block_type() {
            let registry = make_registry();
            let graph = Graph {
                blocks: vec![make_block("b0", "NonExistentBlock123")],
                edges: vec![],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let plugin_diags: Vec<_> = diags
                .iter()
                .filter(|d| d.rule == "missing-plugin")
                .collect();
            assert!(
                !plugin_diags.is_empty(),
                "expected missing plugin diagnostic"
            );
            assert_eq!(plugin_diags[0].severity, Severity::Error);
            assert!(plugin_diags[0].message.contains("NonExistentBlock123"));
        }

        #[test]
        fn test_known_block_type_not_flagged() {
            let registry = make_registry();
            let graph = Graph {
                blocks: vec![make_block("b0", "ReLU")],
                edges: vec![],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let plugin_diags: Vec<_> = diags
                .iter()
                .filter(|d| d.rule == "missing-plugin")
                .collect();
            assert_eq!(plugin_diags.len(), 0);
        }
    }

    mod missing_params {
        use super::*;

        #[test]
        fn test_conv2d_without_kernel() {
            let registry = make_registry();
            // Conv2d requires `kernel` param
            let graph = Graph {
                blocks: vec![make_block("b0", "Conv2d")],
                edges: vec![],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let param_diags: Vec<_> = diags.iter().filter(|d| d.rule == "missing-param").collect();
            assert!(!param_diags.is_empty(), "expected missing param diagnostic");
            assert_eq!(param_diags[0].severity, Severity::Error);
            assert!(param_diags[0].message.contains("kernel"));
        }

        #[test]
        fn test_block_with_all_required_params() {
            let registry = make_registry();
            let mut params = HashMap::new();
            params.insert("kernel".to_string(), p_num(3.0));
            params.insert("out_channels".to_string(), p_num(64.0));
            let graph = Graph {
                blocks: vec![make_block_with_params("b0", "Conv2d", params)],
                edges: vec![],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let param_diags: Vec<_> = diags.iter().filter(|d| d.rule == "missing-param").collect();
            assert_eq!(param_diags.len(), 0);
        }
    }

    mod shapes {
        use super::*;

        #[test]
        fn test_add_shape_mismatch() {
            let registry = make_registry();
            // Add with [3, 32, 32] and [3, 64, 64] should error
            let mut params_a = HashMap::new();
            params_a.insert("dims".to_string(), p_shape(vec![3, 32, 32]));
            let mut params_b = HashMap::new();
            params_b.insert("dims".to_string(), p_shape(vec![3, 64, 64]));
            let graph = Graph {
                blocks: vec![
                    make_block_with_params("b0", "Input", params_a),
                    make_block_with_params("b1", "Input", params_b),
                    make_block("b2", "Add"),
                ],
                edges: vec![make_edge("b0", "b2"), make_edge("b1", "b2")],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let shape_diags: Vec<_> = diags
                .iter()
                .filter(|d| d.rule == "shape-mismatch")
                .collect();
            assert!(
                !shape_diags.is_empty(),
                "expected shape mismatch diagnostic"
            );
            assert_eq!(shape_diags[0].severity, Severity::Error);
        }

        #[test]
        fn test_add_compatible_shapes() {
            let registry = make_registry();
            let mut params = HashMap::new();
            params.insert("dims".to_string(), p_shape(vec![3, 32, 32]));
            let graph = Graph {
                blocks: vec![
                    make_block_with_params("b0", "Input", params.clone()),
                    make_block_with_params("b1", "Input", params),
                    make_block("b2", "Add"),
                ],
                edges: vec![make_edge("b0", "b2"), make_edge("b1", "b2")],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let shape_diags: Vec<_> = diags
                .iter()
                .filter(|d| d.rule == "shape-mismatch")
                .collect();
            assert_eq!(shape_diags.len(), 0);
        }
    }

    mod duplicate_tensor_names {
        use super::*;

        #[test]
        fn test_duplicate_from_different_blocks() {
            let registry = make_registry();
            let mut params = HashMap::new();
            params.insert("dims".to_string(), p_shape(vec![1]));
            let graph = Graph {
                blocks: vec![
                    make_block_with_params("b0", "Input", params),
                    make_block("b1", "ReLU"),
                    make_block("b2", "ReLU"),
                    make_block("b3", "Add"),
                ],
                edges: vec![
                    make_edge_with_name("b0", "b2", "feat"),
                    make_edge_with_name("b1", "b3", "feat"), // same name, different source -> duplicate
                    make_edge("b2", "b3"),
                ],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let dup_diags: Vec<_> = diags
                .iter()
                .filter(|d| d.rule == "duplicate-tensor-name")
                .collect();
            assert!(
                !dup_diags.is_empty(),
                "expected duplicate tensor name diagnostic"
            );
            assert_eq!(dup_diags[0].severity, Severity::Error);
            assert!(dup_diags[0].message.contains("feat"));
        }

        #[test]
        fn test_same_source_fanout_allowed() {
            let registry = make_registry();
            let mut params = HashMap::new();
            params.insert("dims".to_string(), p_shape(vec![1]));
            let graph = Graph {
                blocks: vec![
                    make_block_with_params("b0", "Input", params),
                    make_block("b1", "ReLU"),
                    make_block("b2", "ReLU"),
                    make_block("b3", "Add"),
                ],
                edges: vec![
                    make_edge_with_name("b0", "b1", "feat"),
                    make_edge_with_name("b0", "b2", "feat"), // same source -> valid fan-out
                    make_edge("b1", "b3"),
                    make_edge("b2", "b3"),
                ],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let dup_diags: Vec<_> = diags
                .iter()
                .filter(|d| d.rule == "duplicate-tensor-name")
                .collect();
            assert_eq!(dup_diags.len(), 0);
        }
    }

    mod undefined_tensor_refs {
        use super::*;

        #[test]
        fn test_undefined_tensor_reference() {
            let registry = make_registry();
            // Block with a list param containing a bareword referencing an undefined tensor
            let mut params = HashMap::new();
            params.insert(
                "sources".to_string(),
                p_list(vec![p_bareword("nonexistent_tensor")]),
            );
            let graph = Graph {
                blocks: vec![make_block_with_params("b0", "Add", params)],
                edges: vec![],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let ref_diags: Vec<_> = diags
                .iter()
                .filter(|d| d.rule == "undefined-tensor-ref")
                .collect();
            assert!(
                !ref_diags.is_empty(),
                "expected undefined tensor ref diagnostic"
            );
            assert_eq!(ref_diags[0].severity, Severity::Error);
            assert!(ref_diags[0].message.contains("nonexistent_tensor"));
        }

        #[test]
        fn test_defined_tensor_not_flagged() {
            let registry = make_registry();
            let mut params = HashMap::new();
            params.insert("dims".to_string(), p_shape(vec![3, 32, 32]));
            let mut join_params = HashMap::new();
            join_params.insert("sources".to_string(), p_list(vec![p_bareword("my_tensor")]));
            let graph = Graph {
                blocks: vec![
                    make_block_with_params("b0", "Input", params),
                    make_block_with_params("b1", "ReLU", join_params),
                ],
                edges: vec![make_edge_with_name("b0", "b1", "my_tensor")],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let ref_diags: Vec<_> = diags
                .iter()
                .filter(|d| d.rule == "undefined-tensor-ref")
                .collect();
            assert_eq!(ref_diags.len(), 0);
        }
    }

    mod unused_tensors {
        use super::*;

        #[test]
        fn test_unused_named_tensor() {
            let registry = make_registry();
            let mut params = HashMap::new();
            params.insert("dims".to_string(), p_shape(vec![3]));
            let graph = Graph {
                blocks: vec![make_block_with_params("b0", "Input", params)],
                edges: vec![Edge {
                    from: "b0".to_string(),
                    to: "orphan_block".to_string(),
                    tensor_name: Some("orphan".to_string()),
                    shape: None,
                }],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let unused_diags: Vec<_> = diags.iter().filter(|d| d.rule == "unused-tensor").collect();
            assert!(!unused_diags.is_empty(), "expected unused tensor warning");
            assert_eq!(unused_diags[0].severity, Severity::Warning);
            assert!(unused_diags[0].message.contains("orphan"));
        }

        #[test]
        fn test_used_tensor_not_flagged() {
            let registry = make_registry();
            let mut params = HashMap::new();
            params.insert("dims".to_string(), p_shape(vec![3]));
            let graph = Graph {
                blocks: vec![
                    make_block_with_params("b0", "Input", params),
                    make_block("b1", "ReLU"),
                ],
                edges: vec![make_edge_with_name("b0", "b1", "used_tensor")],
                groups: vec![],
            };
            let diags = lint(&graph, &registry);
            let unused_diags: Vec<_> = diags.iter().filter(|d| d.rule == "unused-tensor").collect();
            assert_eq!(unused_diags.len(), 0);
        }
    }
}
