use std::collections::{HashMap, HashSet, VecDeque};

use crate::ast::graph::{Block, Graph};
use crate::ast::graph_utils::topo_sort_ids;
use crate::ast::nodes::{ParamValue, SourceLoc};
use crate::lint::{LintDiagnostic, Severity};
use crate::shape::{infer_shapes, BlockRegistry};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn unknown_loc() -> SourceLoc {
    SourceLoc {
        line: 0,
        col: 0,
        offset: 0,
    }
}

fn block_loc(b: &Block) -> SourceLoc {
    b.loc.clone()
}

// ---------------------------------------------------------------------------
// Rule 1: check_cycles
// ---------------------------------------------------------------------------

/// Cycle detection (must run first). Uses `topo_sort_ids`; if it returns None,
/// find all blocks in cycles via Kahn's algorithm inversion.
pub fn check_cycles(graph: &Graph) -> Vec<LintDiagnostic> {
    let order = topo_sort_ids(&graph.blocks, &graph.edges);
    if order.is_some() {
        return Vec::new();
    }

    // Find all blocks involved in a cycle using Kahn inversion
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut adj_out: HashMap<String, Vec<String>> = HashMap::new();

    for b in &graph.blocks {
        in_degree.insert(b.id.clone(), 0);
        adj_out.insert(b.id.clone(), Vec::new());
    }

    for e in &graph.edges {
        if !in_degree.contains_key(&e.to) || !in_degree.contains_key(&e.from) {
            continue;
        }
        *in_degree.get_mut(&e.to).unwrap() += 1;
        adj_out.get_mut(&e.from).unwrap().push(e.to.clone());
    }

    let mut queue: VecDeque<String> = VecDeque::new();
    for (id, deg) in &in_degree {
        if *deg == 0 {
            queue.push_back(id.clone());
        }
    }

    let mut visited: HashSet<String> = HashSet::new();
    while let Some(id) = queue.pop_front() {
        visited.insert(id.clone());
        if let Some(nexts) = adj_out.get(&id) {
            for next in nexts {
                if let Some(deg) = in_degree.get_mut(next) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(next.clone());
                    }
                }
            }
        }
    }

    graph
        .blocks
        .iter()
        .filter(|b| !visited.contains(&b.id))
        .map(|b| LintDiagnostic {
            severity: Severity::Error,
            message: format!("Block \"{}\" ({}) is part of a cycle", b.block_type, b.id),
            loc: block_loc(b),
            rule: "cycle".to_string(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Rule 2: check_missing_plugins
// ---------------------------------------------------------------------------

/// Missing plugin/block type: unknown type not in registry.
pub fn check_missing_plugins(graph: &Graph, registry: &BlockRegistry) -> Vec<LintDiagnostic> {
    let mut diags = Vec::new();
    for b in &graph.blocks {
        if registry.get(&b.block_type).is_none() {
            diags.push(LintDiagnostic {
                severity: Severity::Error,
                message: format!("Unknown block type: \"{}\"", b.block_type),
                loc: block_loc(b),
                rule: "missing-plugin".to_string(),
            });
        }
    }
    diags
}

// ---------------------------------------------------------------------------
// Rule 3: check_missing_params
// ---------------------------------------------------------------------------

/// Missing required params: for each block with a registered BlockDef, check
/// all required params are present.
pub fn check_missing_params(graph: &Graph, registry: &BlockRegistry) -> Vec<LintDiagnostic> {
    let mut diags = Vec::new();
    for b in &graph.blocks {
        let def = match registry.get(&b.block_type) {
            Some(def) => def,
            None => continue,
        };
        for spec in def.params() {
            if spec.required && !b.params.contains_key(&spec.name) {
                diags.push(LintDiagnostic {
                    severity: Severity::Error,
                    message: format!(
                        "Block \"{}\" ({}) is missing required param \"{}\"",
                        b.block_type, b.id, spec.name
                    ),
                    loc: block_loc(b),
                    rule: "missing-param".to_string(),
                });
            }
        }
    }
    diags
}

// ---------------------------------------------------------------------------
// Rule 4: check_shapes
// ---------------------------------------------------------------------------

/// Shape compatibility: runs `infer_shapes`, collects errors, and checks
/// elementwise merge compatibility (all inputs to Add/Mul/Sub/Div must have
/// matching shapes).
pub fn check_shapes(graph: &Graph, registry: &BlockRegistry) -> Vec<LintDiagnostic> {
    let result = infer_shapes(graph, registry);
    let block_map: HashMap<&str, &Block> =
        graph.blocks.iter().map(|b| (b.id.as_str(), b)).collect();

    let mut diags = Vec::new();

    // Collect errors from infer_shapes
    for e in &result.errors {
        if e.message == "Cycle detected in graph" {
            continue; // covered by check_cycles
        }
        let b = e.block_id.as_str();
        let loc = block_map
            .get(b)
            .map(|blk| block_loc(blk))
            .unwrap_or_else(unknown_loc);
        diags.push(LintDiagnostic {
            severity: Severity::Error,
            message: e.message.clone(),
            loc,
            rule: "shape-mismatch".to_string(),
        });
    }

    // Elementwise merge compatibility check:
    // For blocks where the output shape equals the first input shape (passthrough
    // semantics, e.g., Add/Mul/Sub/Div), verify all input shapes match.
    let inferred_block_map: HashMap<&str, &Block> = result
        .graph
        .blocks
        .iter()
        .map(|b| (b.id.as_str(), b))
        .collect();

    for b in &graph.blocks {
        let def = registry.get(&b.block_type);
        if def.is_none() {
            continue;
        }
        let inferred = match inferred_block_map.get(b.id.as_str()) {
            Some(blk) => blk,
            None => continue,
        };
        let input_shapes = &inferred.input_shapes;
        if input_shapes.len() < 2 {
            continue;
        }
        let output_shapes = &inferred.output_shapes;
        if output_shapes.len() == 1 {
            let out = &output_shapes[0];
            let first_in = &input_shapes[0];
            if out.len() == first_in.len() && out.iter().zip(first_in.iter()).all(|(a, b)| a == b) {
                // Output matches first input — this may be an elementwise op.
                // Verify all inputs match the first.
                for i in 1..input_shapes.len() {
                    let inp = &input_shapes[i];
                    let mismatch = inp.len() != first_in.len()
                        || inp.iter().zip(first_in.iter()).any(|(a, b)| a != b);
                    if mismatch {
                        let first_str: Vec<String> =
                            first_in.iter().map(|d| d.to_string()).collect();
                        let inp_str: Vec<String> = inp.iter().map(|d| d.to_string()).collect();
                        diags.push(LintDiagnostic {
                            severity: Severity::Error,
                            message: format!(
                                "Block \"{}\" ({}): input shape [{}] does not match expected [{}]",
                                b.block_type,
                                b.id,
                                inp_str.join(","),
                                first_str.join(",")
                            ),
                            loc: block_loc(b),
                            rule: "shape-mismatch".to_string(),
                        });
                    }
                }
            }
        }
    }

    diags
}

// ---------------------------------------------------------------------------
// Rule 5: check_duplicate_tensor_names
// ---------------------------------------------------------------------------

/// Duplicate tensor names: error when the same tensor name is produced by
/// different source blocks (fan-out from the same block is allowed).
pub fn check_duplicate_tensor_names(graph: &Graph) -> Vec<LintDiagnostic> {
    // Track edges by (tensorName, source block)
    let mut seen: HashMap<String, Vec<String>> = HashMap::new();
    for e in &graph.edges {
        if let Some(ref name) = e.tensor_name {
            seen.entry(name.clone()).or_default().push(e.from.clone());
        }
    }

    let mut diags = Vec::new();
    for (name, sources) in &seen {
        let unique_sources: HashSet<&str> = sources.iter().map(|s| s.as_str()).collect();
        if unique_sources.len() > 1 {
            diags.push(LintDiagnostic {
                severity: Severity::Error,
                message: format!(
                    "Tensor name \"{}\" is defined by {} different blocks",
                    name,
                    unique_sources.len()
                ),
                loc: unknown_loc(),
                rule: "duplicate-tensor-name".to_string(),
            });
        }
    }

    diags
}

// ---------------------------------------------------------------------------
// Rule 6: check_undefined_tensor_refs
// ---------------------------------------------------------------------------

/// Undefined tensor references: check if any tensor name referenced in joins
/// is never defined. Scans block params of list kind for bareword items; if
/// the bareword is not a defined tensor name (from edge tensor_names), it is
/// flagged as an undefined reference.
pub fn check_undefined_tensor_refs(graph: &Graph) -> Vec<LintDiagnostic> {
    // Collect all defined tensor names from edges
    let defined: HashSet<&str> = graph
        .edges
        .iter()
        .filter_map(|e| e.tensor_name.as_deref())
        .collect();

    let mut diags = Vec::new();

    for b in &graph.blocks {
        for (_param_name, pv) in &b.params {
            if let ParamValue::List(list) = pv {
                for item in &list.items {
                    if let ParamValue::Bareword(bw) = item {
                        if !defined.contains(bw.value.as_str()) {
                            diags.push(LintDiagnostic {
                                severity: Severity::Error,
                                message: format!("Undefined tensor reference: \"{}\"", bw.value),
                                loc: block_loc(b),
                                rule: "undefined-tensor-ref".to_string(),
                            });
                        }
                    }
                }
            }
        }
    }

    diags
}

// ---------------------------------------------------------------------------
// Rule 7: check_unused_tensors
// ---------------------------------------------------------------------------

/// Unused named tensors: named tensors (edges with tensorName) whose target
/// block does not exist in the graph produce warnings.
pub fn check_unused_tensors(graph: &Graph) -> Vec<LintDiagnostic> {
    let block_ids: HashSet<&str> = graph.blocks.iter().map(|b| b.id.as_str()).collect();

    let mut diags = Vec::new();
    for e in &graph.edges {
        if let Some(ref name) = e.tensor_name {
            if !block_ids.contains(e.to.as_str()) {
                diags.push(LintDiagnostic {
                    severity: Severity::Warning,
                    message: format!("Named tensor \"{}\" is created but never consumed", name),
                    loc: unknown_loc(),
                    rule: "unused-tensor".to_string(),
                });
            }
        }
    }

    diags
}

// =========================================================================
// Unit tests for each rule
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::graph::{Block, Edge, Shape};
    use crate::ast::nodes::{BarewordVal, ListVal, NumberVal, ShapeVal};
    use crate::block::types::{BlockDef, ParamSpec, ParamType};

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

    // ------------------------------------------------------------------
    // Mock BlockDef implementations
    // ------------------------------------------------------------------

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
            _inputs: &[Shape],
            params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<Shape>, String> {
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
            inputs: &[Shape],
            _params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<Shape>, String> {
            Ok(inputs.to_vec())
        }
    }

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
            inputs: &[Shape],
            _params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<Shape>, String> {
            if inputs.is_empty() {
                return Err("Add requires inputs".to_string());
            }
            Ok(vec![inputs[0].clone()])
        }
    }

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
            _inputs: &[Shape],
            params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<Shape>, String> {
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
    // Rule: check_cycles
    // ==============================================================

    #[test]
    fn test_check_cycles_no_cycle() {
        let graph = Graph {
            blocks: vec![make_block("a", "ReLU"), make_block("b", "ReLU")],
            edges: vec![make_edge("a", "b")],
            groups: vec![],
        };
        assert!(check_cycles(&graph).is_empty());
    }

    #[test]
    fn test_check_cycles_simple() {
        let graph = Graph {
            blocks: vec![make_block("a", "ReLU"), make_block("b", "ReLU")],
            edges: vec![make_edge("a", "b"), make_edge("b", "a")],
            groups: vec![],
        };
        let diags = check_cycles(&graph);
        assert_eq!(diags.len(), 2);
        for d in &diags {
            assert_eq!(d.rule, "cycle");
            assert_eq!(d.severity, Severity::Error);
        }
    }

    #[test]
    fn test_check_cycles_three() {
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
        let diags = check_cycles(&graph);
        assert_eq!(diags.len(), 3);
    }

    // ==============================================================
    // Rule: check_missing_plugins
    // ==============================================================

    #[test]
    fn test_check_missing_plugins_unknown() {
        let registry = make_registry();
        let graph = Graph {
            blocks: vec![make_block("b0", "NonExistentBlock")],
            edges: vec![],
            groups: vec![],
        };
        let diags = check_missing_plugins(&graph, &registry);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].rule, "missing-plugin");
        assert!(diags[0].message.contains("NonExistentBlock"));
    }

    #[test]
    fn test_check_missing_plugins_known() {
        let registry = make_registry();
        let graph = Graph {
            blocks: vec![make_block("b0", "ReLU")],
            edges: vec![],
            groups: vec![],
        };
        let diags = check_missing_plugins(&graph, &registry);
        assert_eq!(diags.len(), 0);
    }

    // ==============================================================
    // Rule: check_missing_params
    // ==============================================================

    #[test]
    fn test_check_missing_params_missing() {
        let registry = make_registry();
        // Conv2d without kernel
        let graph = Graph {
            blocks: vec![make_block("b0", "Conv2d")],
            edges: vec![],
            groups: vec![],
        };
        let diags = check_missing_params(&graph, &registry);
        assert!(!diags.is_empty());
        assert!(diags.iter().any(|d| d.message.contains("kernel")));
    }

    #[test]
    fn test_check_missing_params_present() {
        let registry = make_registry();
        let mut params = HashMap::new();
        params.insert("kernel".to_string(), p_num(3.0));
        params.insert("out_channels".to_string(), p_num(64.0));
        let graph = Graph {
            blocks: vec![make_block_with_params("b0", "Conv2d", params)],
            edges: vec![],
            groups: vec![],
        };
        let diags = check_missing_params(&graph, &registry);
        assert_eq!(diags.len(), 0);
    }

    #[test]
    fn test_check_missing_params_unknown_block_skipped() {
        let registry = make_registry();
        // Unknown block type should be skipped (no BlockDef to check params against)
        let graph = Graph {
            blocks: vec![make_block("b0", "NonExistent")],
            edges: vec![],
            groups: vec![],
        };
        let diags = check_missing_params(&graph, &registry);
        assert_eq!(diags.len(), 0);
    }

    // ==============================================================
    // Rule: check_shapes
    // ==============================================================

    #[test]
    fn test_check_shapes_add_mismatch() {
        let registry = make_registry();
        let mut pa = HashMap::new();
        pa.insert("dims".to_string(), p_shape(vec![3, 32, 32]));
        let mut pb = HashMap::new();
        pb.insert("dims".to_string(), p_shape(vec![3, 64, 64]));
        let graph = Graph {
            blocks: vec![
                make_block_with_params("b0", "Input", pa),
                make_block_with_params("b1", "Input", pb),
                make_block("b2", "Add"),
            ],
            edges: vec![make_edge("b0", "b2"), make_edge("b1", "b2")],
            groups: vec![],
        };
        let diags = check_shapes(&graph, &registry);
        let shape_diags: Vec<_> = diags
            .iter()
            .filter(|d| d.rule == "shape-mismatch")
            .collect();
        assert!(!shape_diags.is_empty(), "expected shape mismatch diags");
    }

    #[test]
    fn test_check_shapes_add_compatible() {
        let registry = make_registry();
        let mut p = HashMap::new();
        p.insert("dims".to_string(), p_shape(vec![3, 32, 32]));
        let graph = Graph {
            blocks: vec![
                make_block_with_params("b0", "Input", p.clone()),
                make_block_with_params("b1", "Input", p),
                make_block("b2", "Add"),
            ],
            edges: vec![make_edge("b0", "b2"), make_edge("b1", "b2")],
            groups: vec![],
        };
        let diags = check_shapes(&graph, &registry);
        let shape_diags: Vec<_> = diags
            .iter()
            .filter(|d| d.rule == "shape-mismatch")
            .collect();
        assert_eq!(shape_diags.len(), 0);
    }

    // ==============================================================
    // Rule: check_duplicate_tensor_names
    // ==============================================================

    #[test]
    fn test_duplicate_tensor_names_different_sources() {
        let graph = Graph {
            blocks: vec![
                make_block("b0", "ReLU"),
                make_block("b1", "ReLU"),
                make_block("b2", "Add"),
            ],
            edges: vec![
                make_edge_with_name("b0", "b2", "feat"),
                make_edge_with_name("b1", "b2", "feat"),
            ],
            groups: vec![],
        };
        let diags = check_duplicate_tensor_names(&graph);
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("feat"));
    }

    #[test]
    fn test_duplicate_tensor_names_same_source() {
        let graph = Graph {
            blocks: vec![
                make_block("b0", "ReLU"),
                make_block("b1", "ReLU"),
                make_block("b2", "Add"),
            ],
            edges: vec![
                make_edge_with_name("b0", "b1", "feat"),
                make_edge_with_name("b0", "b2", "feat"),
            ],
            groups: vec![],
        };
        let diags = check_duplicate_tensor_names(&graph);
        assert_eq!(diags.len(), 0);
    }

    #[test]
    fn test_duplicate_tensor_names_no_edges() {
        let graph = Graph {
            blocks: vec![],
            edges: vec![],
            groups: vec![],
        };
        let diags = check_duplicate_tensor_names(&graph);
        assert_eq!(diags.len(), 0);
    }

    // ==============================================================
    // Rule: check_undefined_tensor_refs
    // ==============================================================

    #[test]
    fn test_undefined_tensor_refs_undefined() {
        // Block with list param containing bareword referencing undefined tensor
        let mut params = HashMap::new();
        params.insert(
            "sources".to_string(),
            p_list(vec![p_bareword("undefined_name")]),
        );
        let graph = Graph {
            blocks: vec![make_block_with_params("b0", "Add", params)],
            edges: vec![],
            groups: vec![],
        };
        let diags = check_undefined_tensor_refs(&graph);
        assert!(!diags.is_empty());
        assert!(diags[0].message.contains("undefined_name"));
    }

    #[test]
    fn test_undefined_tensor_refs_defined() {
        let mut params = HashMap::new();
        params.insert("sources".to_string(), p_list(vec![p_bareword("my_tensor")]));
        let graph = Graph {
            blocks: vec![make_block_with_params("b0", "Add", params)],
            edges: vec![make_edge_with_name("b0", "b0", "my_tensor")],
            groups: vec![],
        };
        let diags = check_undefined_tensor_refs(&graph);
        assert_eq!(diags.len(), 0);
    }

    #[test]
    fn test_undefined_tensor_refs_no_list_params() {
        let graph = Graph {
            blocks: vec![make_block("b0", "ReLU")],
            edges: vec![],
            groups: vec![],
        };
        let diags = check_undefined_tensor_refs(&graph);
        assert_eq!(diags.len(), 0);
    }

    // ==============================================================
    // Rule: check_unused_tensors
    // ==============================================================

    #[test]
    fn test_unused_tensors_no_downstream() {
        let graph = Graph {
            blocks: vec![make_block("b0", "Input")],
            edges: vec![make_edge_with_name("b0", "nonexistent", "orphan")],
            groups: vec![],
        };
        let diags = check_unused_tensors(&graph);
        assert!(!diags.is_empty());
        assert_eq!(diags[0].severity, Severity::Warning);
        assert!(diags[0].message.contains("orphan"));
    }

    #[test]
    fn test_unused_tensors_has_downstream() {
        let graph = Graph {
            blocks: vec![make_block("b0", "Input"), make_block("b1", "ReLU")],
            edges: vec![make_edge_with_name("b0", "b1", "active")],
            groups: vec![],
        };
        let diags = check_unused_tensors(&graph);
        assert_eq!(diags.len(), 0);
    }

    #[test]
    fn test_unused_tensors_no_tensor_names() {
        let graph = Graph {
            blocks: vec![make_block("b0", "Input"), make_block("b1", "ReLU")],
            edges: vec![make_edge("b0", "b1")],
            groups: vec![],
        };
        let diags = check_unused_tensors(&graph);
        assert_eq!(diags.len(), 0);
    }
}
