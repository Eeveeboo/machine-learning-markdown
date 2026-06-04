use std::collections::HashMap;

use crate::ast::graph::{Block, Edge, Graph, Group};
use crate::ast::nodes::{ASTNode, BlockDecl, Param, ParamValue};

// ---------------------------------------------------------------------------
// BuildState — mutable state accumulated during graph construction
// ---------------------------------------------------------------------------

struct BuildState {
    blocks: Vec<Block>,
    edges: Vec<Edge>,
    groups: Vec<Group>,
    /// Counter per block_type for unique ID generation
    type_count: HashMap<String, usize>,
    /// Named tensors: name -> block id
    named_tensors: HashMap<String, String>,
    /// Current tail block id (last block in linear chain)
    current_tail: Option<String>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Convert a slice of Param to a HashMap (equivalent to TS `paramsToRecord`).
fn params_to_record(params: &[Param]) -> HashMap<String, ParamValue> {
    let mut map = HashMap::new();
    for p in params {
        map.insert(p.name.clone(), p.value.clone());
    }
    map
}

/// Generate a unique block id like `"Conv2d_0"`, `"Conv2d_1"`, etc.
fn make_block_id(state: &mut BuildState, block_type: &str) -> String {
    let count = state.type_count.entry(block_type.to_string()).or_insert(0);
    let id = format!("{}_{}", block_type, *count);
    *count += 1;
    id
}

/// Create a Block from a BlockDecl, push it to state, register in groups,
/// and return the new block's id.
fn add_block(state: &mut BuildState, decl: &BlockDecl, group_stack: &[usize]) -> String {
    let id = make_block_id(state, &decl.block_type);
    let block = Block {
        id: id.clone(),
        block_type: decl.block_type.clone(),
        params: params_to_record(&decl.params),
        input_shapes: vec![],
        output_shapes: vec![],
        param_count: None,
        show_depth: None,
        loc: decl.loc.clone(),
    };
    state.blocks.push(block);
    // Register this block id in all ancestor groups
    for &grp_idx in group_stack {
        state.groups[grp_idx].block_ids.push(id.clone());
    }
    id
}

// ---------------------------------------------------------------------------
// Recursive AST walker
// ---------------------------------------------------------------------------

fn process_nodes(state: &mut BuildState, nodes: &[ASTNode], group_stack: &mut Vec<usize>) {
    for node in nodes {
        match node {
            ASTNode::Comment(_) => {
                // skip
            }

            ASTNode::Block(decl) => {
                let block_id = add_block(state, decl, group_stack);
                // Wire edge from current tail (linear chain)
                if let Some(ref tail) = state.current_tail {
                    state.edges.push(Edge {
                        from: tail.clone(),
                        to: block_id.clone(),
                        tensor_name: None,
                        shape: None,
                    });
                }
                state.current_tail = Some(block_id);
            }

            ASTNode::TensorName(tn) => {
                // Name the current tail for each name provided
                if let Some(ref tail) = state.current_tail {
                    for name in &tn.names {
                        state.named_tensors.insert(name.clone(), tail.clone());
                    }
                }
            }

            ASTNode::TensorJoin(tj) => {
                let block_id = add_block(state, &tj.target, group_stack);
                // Wire edges from each named source
                for src in &tj.sources {
                    if let Some(from_id) = state.named_tensors.get(src) {
                        state.edges.push(Edge {
                            from: from_id.clone(),
                            to: block_id.clone(),
                            tensor_name: Some(src.clone()),
                            shape: None,
                        });
                    }
                }
                state.current_tail = Some(block_id);
            }

            ASTNode::Group(g) => {
                // Create a new Group and push it to state; record its index
                // so we can push block ids into it during recursive processing.
                let group_idx = state.groups.len();
                state.groups.push(Group {
                    path: g.path.clone(),
                    block_ids: Vec::new(),
                });
                group_stack.push(group_idx);
                process_nodes(state, &g.body, group_stack);
                group_stack.pop();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a `Graph` from a flat list of AST nodes produced by the parser.
/// This is the Rust equivalent of the TypeScript `buildGraph` in
/// `src/parser/build-graph.ts`.
pub fn build_graph(nodes: &[ASTNode]) -> Graph {
    let mut state = BuildState {
        blocks: Vec::new(),
        edges: Vec::new(),
        groups: Vec::new(),
        type_count: HashMap::new(),
        named_tensors: HashMap::new(),
        current_tail: None,
    };

    let mut group_stack: Vec<usize> = Vec::new();
    process_nodes(&mut state, nodes, &mut group_stack);

    // Cleanup: remove spurious chain edges between Input blocks
    state.edges.retain(|e| {
        let from_is_input = state
            .blocks
            .iter()
            .any(|b| b.id == e.from && b.block_type == "Input");
        let to_is_input = state
            .blocks
            .iter()
            .any(|b| b.id == e.to && b.block_type == "Input");
        !(from_is_input && to_is_input)
    });

    Graph {
        blocks: state.blocks,
        edges: state.edges,
        groups: state.groups,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::nodes::*;

    // ---- helpers -----------------------------------------------------------

    fn dummy_loc() -> SourceLoc {
        SourceLoc {
            line: 0,
            col: 0,
            offset: 0,
        }
    }

    fn make_block_decl(block_type: &str, params: Vec<Param>) -> Box<BlockDecl> {
        Box::new(BlockDecl::new(block_type.to_string(), params, dummy_loc()))
    }

    fn make_param(name: &str, value: ParamValue) -> Param {
        Param {
            name: name.to_string(),
            value,
            loc: dummy_loc(),
        }
    }

    fn make_number_val(v: f64) -> ParamValue {
        ParamValue::Number(Box::new(NumberVal::new(v, dummy_loc())))
    }

    fn make_shape_val(dims: Vec<usize>) -> ParamValue {
        ParamValue::Shape(Box::new(ShapeVal::new(dims, dummy_loc())))
    }

    fn make_tensor_name(names: Vec<&str>) -> Box<TensorName> {
        Box::new(TensorName::new(
            names.iter().map(|s| s.to_string()).collect(),
            dummy_loc(),
        ))
    }

    fn make_tensor_join(sources: Vec<&str>, target: &str) -> Box<TensorJoin> {
        Box::new(TensorJoin::new(
            sources.iter().map(|s| s.to_string()).collect(),
            BlockDecl::new(target.to_string(), vec![], dummy_loc()),
            dummy_loc(),
        ))
    }

    fn make_group(path: Vec<&str>, body: Vec<ASTNode>) -> Box<GroupDecl> {
        Box::new(GroupDecl::new(
            path.iter().map(|s| s.to_string()).collect(),
            body,
            dummy_loc(),
        ))
    }

    fn count_block_type(graph: &Graph, block_type: &str) -> usize {
        graph
            .blocks
            .iter()
            .filter(|b| b.block_type == block_type)
            .count()
    }

    fn find_block<'a>(graph: &'a Graph, id: &str) -> Option<&'a Block> {
        graph.blocks.iter().find(|b| b.id == id)
    }

    // ---- tests -------------------------------------------------------------

    /// Simple linear chain: Input -> Conv2d -> Output
    #[test]
    fn test_simple_linear_chain() {
        let nodes = vec![
            ASTNode::Block(make_block_decl("Input", vec![])),
            ASTNode::Block(make_block_decl("Conv2d", vec![])),
            ASTNode::Block(make_block_decl("Output", vec![])),
        ];

        let graph = build_graph(&nodes);

        assert_eq!(graph.blocks.len(), 3);
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.groups.len(), 0);

        // Check block ids
        assert_eq!(graph.blocks[0].id, "Input_0");
        assert_eq!(graph.blocks[1].id, "Conv2d_0");
        assert_eq!(graph.blocks[2].id, "Output_0");

        // Check edges: Input_0 -> Conv2d_0, Conv2d_0 -> Output_0
        assert_eq!(graph.edges[0].from, "Input_0");
        assert_eq!(graph.edges[0].to, "Conv2d_0");
        assert_eq!(graph.edges[1].from, "Conv2d_0");
        assert_eq!(graph.edges[1].to, "Output_0");

        // No tensor names on chain edges
        assert!(graph.edges[0].tensor_name.is_none());
        assert!(graph.edges[1].tensor_name.is_none());
    }

    /// Naming tensors: Input(shape=(3,32,32)) -> [features]
    #[test]
    fn test_naming_tensors() {
        let nodes = vec![
            ASTNode::Block(make_block_decl(
                "Input",
                vec![make_param("shape", make_shape_val(vec![3, 32, 32]))],
            )),
            ASTNode::TensorName(make_tensor_name(vec!["features"])),
        ];

        let graph = build_graph(&nodes);

        assert_eq!(graph.blocks.len(), 1);
        assert_eq!(graph.edges.len(), 0);
        assert_eq!(graph.groups.len(), 0);

        let block = &graph.blocks[0];
        assert_eq!(block.id, "Input_0");
        assert_eq!(block.block_type, "Input");
        // Check param was preserved
        match block.params.get("shape") {
            Some(ParamValue::Shape(s)) => assert_eq!(s.dims, vec![3, 32, 32]),
            _ => panic!("expected shape param"),
        }
    }

    /// Tensor joins: [a, b] -> Add()
    /// Requires named tensors "a" and "b" to be set up first.
    #[test]
    fn test_tensor_joins() {
        // Simulate: Input(a) -> [a]   Input(b) -> [b]   [a, b] -> Add
        let nodes = vec![
            ASTNode::Block(make_block_decl("Input", vec![])),
            ASTNode::TensorName(make_tensor_name(vec!["a"])),
            ASTNode::Block(make_block_decl("Input", vec![])),
            ASTNode::TensorName(make_tensor_name(vec!["b"])),
            ASTNode::TensorJoin(make_tensor_join(vec!["a", "b"], "Add")),
        ];

        let graph = build_graph(&nodes);

        // Blocks: Input_0, Input_1, Add_0
        assert_eq!(graph.blocks.len(), 3);
        assert!(find_block(&graph, "Input_0").is_some());
        assert!(find_block(&graph, "Input_1").is_some());
        assert!(find_block(&graph, "Add_0").is_some());

        // Edges: Input_0 -> Input_1 (spurious, removed by cleanup),
        //        Input_0 (a) -> Add_0, Input_1 (b) -> Add_0
        // After cleanup: 2 edges
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.edges[0].from, "Input_0");
        assert_eq!(graph.edges[0].to, "Add_0");
        assert_eq!(graph.edges[0].tensor_name.as_deref(), Some("a"));
        assert_eq!(graph.edges[1].from, "Input_1");
        assert_eq!(graph.edges[1].to, "Add_0");
        assert_eq!(graph.edges[1].tensor_name.as_deref(), Some("b"));

        // current tail should be Add_0 after processing
        // (checked implicitly by the edges)
    }

    /// Groups with nested blocks
    #[test]
    fn test_groups_with_nested_blocks() {
        let body = vec![
            ASTNode::Block(make_block_decl("Conv2d", vec![])),
            ASTNode::Block(make_block_decl("ReLU", vec![])),
        ];
        let nodes = vec![ASTNode::Group(make_group(vec!["Encoder"], body))];

        let graph = build_graph(&nodes);

        assert_eq!(graph.blocks.len(), 2);
        assert_eq!(graph.groups.len(), 1);
        assert_eq!(graph.edges.len(), 1); // Conv2d_0 -> ReLU_0

        // Check group
        let group = &graph.groups[0];
        assert_eq!(group.path, vec!["Encoder"]);
        assert_eq!(group.block_ids.len(), 2);
        assert!(group.block_ids.contains(&"Conv2d_0".to_string()));
        assert!(group.block_ids.contains(&"ReLU_0".to_string()));

        // Check blocks have correct ids
        assert_eq!(count_block_type(&graph, "Conv2d"), 1);
        assert_eq!(count_block_type(&graph, "ReLU"), 1);
    }

    /// Multiple chains
    #[test]
    fn test_multiple_chains() {
        let nodes = vec![
            // Chain 1: A -> B -> C
            ASTNode::Block(make_block_decl("A", vec![])),
            ASTNode::Block(make_block_decl("B", vec![])),
            ASTNode::Block(make_block_decl("C", vec![])),
        ];

        let graph = build_graph(&nodes);

        assert_eq!(graph.blocks.len(), 3);
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.edges[0].from, "A_0");
        assert_eq!(graph.edges[0].to, "B_0");
        assert_eq!(graph.edges[1].from, "B_0");
        assert_eq!(graph.edges[1].to, "C_0");
    }

    /// Input->Input edge cleanup
    #[test]
    fn test_input_input_edge_cleanup() {
        let nodes = vec![
            ASTNode::Block(make_block_decl("Input", vec![])),
            ASTNode::Block(make_block_decl("Input", vec![])),
            ASTNode::Block(make_block_decl("Conv2d", vec![])),
        ];

        let graph = build_graph(&nodes);

        // Blocks: Input_0, Input_1, Conv2d_0
        // Edges before cleanup: Input_0 -> Input_1 (spurious), Input_1 -> Conv2d_0
        // After cleanup: only Input_1 -> Conv2d_0
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].from, "Input_1");
        assert_eq!(graph.edges[0].to, "Conv2d_0");
    }

    /// Empty input
    #[test]
    fn test_empty_input() {
        let nodes = vec![];
        let graph = build_graph(&nodes);
        assert_eq!(graph.blocks.len(), 0);
        assert_eq!(graph.edges.len(), 0);
        assert_eq!(graph.groups.len(), 0);
    }

    /// LeNet5 full model — construct from parsed-like AST nodes
    #[test]
    fn test_lenet5_full_model() {
        // Construct a simplified LeNet-5:
        // Input(shape=(1,28,28)) -> Conv2d(kernel_size=5,filters=6) -> Tanh
        //   -> AvgPool(kernel_size=2,stride=2)
        //   -> Conv2d(kernel_size=5,filters=16) -> Tanh
        //   -> AvgPool(kernel_size=2,stride=2)
        //   -> Flatten -> Linear(out_features=120) -> Tanh
        //   -> Linear(out_features=84) -> Tanh
        //   -> Linear(out_features=10) -> Softmax
        let nodes = vec![
            ASTNode::Block(make_block_decl(
                "Input",
                vec![make_param("shape", make_shape_val(vec![1, 28, 28]))],
            )),
            ASTNode::Block(make_block_decl(
                "Conv2d",
                vec![
                    make_param("kernel_size", make_number_val(5.0)),
                    make_param("filters", make_number_val(6.0)),
                ],
            )),
            ASTNode::Block(make_block_decl("Tanh", vec![])),
            ASTNode::Block(make_block_decl(
                "AvgPool",
                vec![
                    make_param("kernel_size", make_number_val(2.0)),
                    make_param("stride", make_number_val(2.0)),
                ],
            )),
            ASTNode::Block(make_block_decl(
                "Conv2d",
                vec![
                    make_param("kernel_size", make_number_val(5.0)),
                    make_param("filters", make_number_val(16.0)),
                ],
            )),
            ASTNode::Block(make_block_decl("Tanh", vec![])),
            ASTNode::Block(make_block_decl(
                "AvgPool",
                vec![
                    make_param("kernel_size", make_number_val(2.0)),
                    make_param("stride", make_number_val(2.0)),
                ],
            )),
            ASTNode::Block(make_block_decl("Flatten", vec![])),
            ASTNode::Block(make_block_decl(
                "Linear",
                vec![make_param("out_features", make_number_val(120.0))],
            )),
            ASTNode::Block(make_block_decl("Tanh", vec![])),
            ASTNode::Block(make_block_decl(
                "Linear",
                vec![make_param("out_features", make_number_val(84.0))],
            )),
            ASTNode::Block(make_block_decl("Tanh", vec![])),
            ASTNode::Block(make_block_decl(
                "Linear",
                vec![make_param("out_features", make_number_val(10.0))],
            )),
            ASTNode::Block(make_block_decl("Softmax", vec![])),
        ];

        let graph = build_graph(&nodes);

        // Count blocks by type
        assert_eq!(count_block_type(&graph, "Input"), 1);
        assert_eq!(count_block_type(&graph, "Conv2d"), 2);
        assert_eq!(count_block_type(&graph, "Tanh"), 4);
        assert_eq!(count_block_type(&graph, "AvgPool"), 2);
        assert_eq!(count_block_type(&graph, "Flatten"), 1);
        assert_eq!(count_block_type(&graph, "Linear"), 3);
        assert_eq!(count_block_type(&graph, "Softmax"), 1);
        assert_eq!(graph.blocks.len(), 14);

        // Edges: 13 edges (14 blocks in a chain)
        assert_eq!(graph.edges.len(), 13);

        // Verify chain order via edges
        let expected_chain = [
            "Input_0",
            "Conv2d_0",
            "Tanh_0",
            "AvgPool_0",
            "Conv2d_1",
            "Tanh_1",
            "AvgPool_1",
            "Flatten_0",
            "Linear_0",
            "Tanh_2",
            "Linear_1",
            "Tanh_3",
            "Linear_2",
            "Softmax_0",
        ];
        for (i, id) in expected_chain.iter().enumerate() {
            assert!(find_block(&graph, id).is_some(), "missing block {}", id);
            if i > 0 {
                // Check edge from previous to this
                assert!(
                    graph
                        .edges
                        .iter()
                        .any(|e| e.from == expected_chain[i - 1] && e.to == *id),
                    "missing edge {} -> {}",
                    expected_chain[i - 1],
                    id
                );
            }
        }

        // No groups
        assert_eq!(graph.groups.len(), 0);

        // Check params on first Conv2d
        let conv1 = find_block(&graph, "Conv2d_0").unwrap();
        match conv1.params.get("kernel_size") {
            Some(ParamValue::Number(v)) => assert_eq!(v.value, 5.0),
            _ => panic!("expected kernel_size=5"),
        }
        match conv1.params.get("filters") {
            Some(ParamValue::Number(v)) => assert_eq!(v.value, 6.0),
            _ => panic!("expected filters=6"),
        }
    }

    /// ResNet bottleneck pattern (fork + join)
    #[test]
    fn test_fork_join_pattern() {
        // Simulate:
        // Input -> Conv2d(kernel=1) -> ReLU -> [skip]
        // Input -> Conv2d(kernel=3) -> ReLU -> [main]
        // [skip, main] -> Add -> Output
        let nodes = vec![
            // Chain 1
            ASTNode::Block(make_block_decl("Input", vec![])),
            ASTNode::Block(make_block_decl(
                "Conv2d",
                vec![make_param("kernel", make_number_val(1.0))],
            )),
            ASTNode::Block(make_block_decl("ReLU", vec![])),
            ASTNode::TensorName(make_tensor_name(vec!["skip"])),
            // Chain 2
            ASTNode::Block(make_block_decl("Input", vec![])),
            ASTNode::Block(make_block_decl(
                "Conv2d",
                vec![make_param("kernel", make_number_val(3.0))],
            )),
            ASTNode::Block(make_block_decl("ReLU", vec![])),
            ASTNode::TensorName(make_tensor_name(vec!["main"])),
            // Join
            ASTNode::TensorJoin(make_tensor_join(vec!["skip", "main"], "Add")),
            // Chain continuation
            ASTNode::Block(make_block_decl("Output", vec![])),
        ];

        let graph = build_graph(&nodes);

        // Total blocks: 2 Inputs, 2 Conv2d, 2 ReLU, 1 Add, 1 Output = 8
        assert_eq!(graph.blocks.len(), 8);
        assert_eq!(count_block_type(&graph, "Input"), 2);
        assert_eq!(count_block_type(&graph, "Conv2d"), 2);
        assert_eq!(count_block_type(&graph, "ReLU"), 2);
        assert_eq!(count_block_type(&graph, "Add"), 1);
        assert_eq!(count_block_type(&graph, "Output"), 1);

        // Edges after Input->Input cleanup:
        // Chain 1: Input_0 -> Conv2d_0 -> ReLU_0
        // (spurious) ReLU_0 -> Input_1 (chain 1 tail to chain 2 first block)
        // Chain 2: Input_1 -> Conv2d_1 -> ReLU_1
        // Join:    ReLU_0(skip) -> Add_0, ReLU_1(main) -> Add_0
        // Chain 3: Add_0 -> Output_0
        // Total: 3 + 1(spurious) + 2 + 1 = 8 edges
        assert_eq!(graph.edges.len(), 8);

        // Verify specific edges
        assert!(graph
            .edges
            .iter()
            .any(|e| e.from == "Input_0" && e.to == "Conv2d_0"));
        assert!(graph
            .edges
            .iter()
            .any(|e| e.from == "Conv2d_0" && e.to == "ReLU_0"));
        // Spurious edge between chains (this is how the TS graph builder works)
        assert!(graph
            .edges
            .iter()
            .any(|e| e.from == "ReLU_0" && e.to == "Input_1"));
        assert!(graph
            .edges
            .iter()
            .any(|e| e.from == "Input_1" && e.to == "Conv2d_1"));
        assert!(graph
            .edges
            .iter()
            .any(|e| e.from == "Conv2d_1" && e.to == "ReLU_1"));

        // Join edges with tensor names
        let join_edge_skip = graph
            .edges
            .iter()
            .find(|e| e.tensor_name.as_deref() == Some("skip"))
            .expect("missing skip edge");
        assert_eq!(join_edge_skip.from, "ReLU_0");
        assert_eq!(join_edge_skip.to, "Add_0");

        let join_edge_main = graph
            .edges
            .iter()
            .find(|e| e.tensor_name.as_deref() == Some("main"))
            .expect("missing main edge");
        assert_eq!(join_edge_main.from, "ReLU_1");
        assert_eq!(join_edge_main.to, "Add_0");

        // Chain from Add to Output
        assert!(graph
            .edges
            .iter()
            .any(|e| e.from == "Add_0" && e.to == "Output_0"));

        // No Input->Input edges
        assert!(!graph.edges.iter().any(|e| {
            let from_is_input = graph
                .blocks
                .iter()
                .any(|b| b.id == e.from && b.block_type == "Input");
            let to_is_input = graph
                .blocks
                .iter()
                .any(|b| b.id == e.to && b.block_type == "Input");
            from_is_input && to_is_input
        }));

        // Check params on Conv2ds
        let conv_1 = find_block(&graph, "Conv2d_0").unwrap();
        match conv_1.params.get("kernel") {
            Some(ParamValue::Number(v)) => assert_eq!(v.value, 1.0),
            _ => panic!("expected kernel=1"),
        }
        let conv_3 = find_block(&graph, "Conv2d_1").unwrap();
        match conv_3.params.get("kernel") {
            Some(ParamValue::Number(v)) => assert_eq!(v.value, 3.0),
            _ => panic!("expected kernel=3"),
        }
    }

    /// Nested groups
    #[test]
    fn test_nested_groups() {
        // [[ Outer ]]
        //   [[ Inner ]]
        //     Conv2d
        //     ReLU
        let inner_body = vec![
            ASTNode::Block(make_block_decl("Conv2d", vec![])),
            ASTNode::Block(make_block_decl("ReLU", vec![])),
        ];
        let outer_body = vec![ASTNode::Group(make_group(vec!["Inner"], inner_body))];
        let nodes = vec![ASTNode::Group(make_group(vec!["Outer"], outer_body))];

        let graph = build_graph(&nodes);

        assert_eq!(graph.blocks.len(), 2);
        assert_eq!(graph.groups.len(), 2);
        // Edges: Conv2d_0 -> ReLU_0
        assert_eq!(graph.edges.len(), 1);

        // Outer group should contain both blocks and the inner group's block_ids should contain both
        let outer = graph
            .groups
            .iter()
            .find(|g| g.path == vec!["Outer"])
            .unwrap();
        let inner = graph
            .groups
            .iter()
            .find(|g| g.path == vec!["Inner"])
            .unwrap();

        // Both blocks belong to inner group
        assert_eq!(inner.block_ids.len(), 2);
        assert!(inner.block_ids.contains(&"Conv2d_0".to_string()));
        assert!(inner.block_ids.contains(&"ReLU_0".to_string()));

        // Both blocks also belong to outer group (via the stack propagation)
        assert_eq!(outer.block_ids.len(), 2);
        assert!(outer.block_ids.contains(&"Conv2d_0".to_string()));
        assert!(outer.block_ids.contains(&"ReLU_0".to_string()));
    }

    #[test]
    fn test_multiple_block_type_counts() {
        let nodes = vec![
            ASTNode::Block(make_block_decl("Conv2d", vec![])),
            ASTNode::Block(make_block_decl("ReLU", vec![])),
            ASTNode::Block(make_block_decl("Conv2d", vec![])),
            ASTNode::Block(make_block_decl("Conv2d", vec![])),
            ASTNode::Block(make_block_decl("ReLU", vec![])),
        ];

        let graph = build_graph(&nodes);

        assert_eq!(graph.blocks.len(), 5);
        assert_eq!(graph.blocks[0].id, "Conv2d_0");
        assert_eq!(graph.blocks[1].id, "ReLU_0");
        assert_eq!(graph.blocks[2].id, "Conv2d_1");
        assert_eq!(graph.blocks[3].id, "Conv2d_2");
        assert_eq!(graph.blocks[4].id, "ReLU_1");
    }
}
