use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::graph::{Block, Edge, Graph, Shape};
use crate::ast::graph_utils::topo_sort_ids;
use crate::block::types::BlockDef;

// ---------------------------------------------------------------------------
// ShapeError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeError {
    pub block_id: String,
    pub message: String,
}

// ---------------------------------------------------------------------------
// ShapeResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ShapeResult {
    pub graph: Graph,
    pub errors: Vec<ShapeError>,
}

// ---------------------------------------------------------------------------
// BlockRegistry
// ---------------------------------------------------------------------------

/// A registry of block definitions for shape inference.
/// This is separate from the global registry to allow testing with mock blocks.
pub struct BlockRegistry {
    blocks: HashMap<String, Arc<dyn BlockDef>>,
}

impl BlockRegistry {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
        }
    }

    pub fn register(&mut self, def: Box<dyn BlockDef>) {
        let name = def.name().to_string();
        self.blocks.insert(name, Arc::from(def));
    }

    /// Register a block definition from an `Arc` (used when cloning from global registry).
    pub fn register_arc(&mut self, def: Arc<dyn BlockDef>) {
        let name = def.name().to_string();
        self.blocks.insert(name, def);
    }

    pub fn get(&self, name: &str) -> Option<&dyn BlockDef> {
        self.blocks.get(name).map(|b| b.as_ref())
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// infer_shapes
// ---------------------------------------------------------------------------

/// Infer shapes through the graph in topological order.
///
/// Returns a new graph (copy) with `output_shapes` on blocks and `shape` on
/// edges filled in, along with any errors encountered.
///
/// The function handles:
/// - Multi-output blocks (e.g., LSTM returns 2 output shapes)
/// - Unknown block types (registers an error, sets output shapes to empty)
/// - Blocks whose `infer_shape` returns an error (caught and registered)
/// - Cycle detection (returns early with an error)
/// - Shape propagation to edges (first output shape is assigned to edges)
pub fn infer_shapes(graph: &Graph, registry: &BlockRegistry) -> ShapeResult {
    let mut errors: Vec<ShapeError> = Vec::new();

    // Deep-copy blocks and edges to avoid mutating the original
    let mut blocks: Vec<Block> = graph.blocks.clone();
    let mut edges: Vec<Edge> = graph.edges.clone();

    // Reset shape fields on the copy
    for block in &mut blocks {
        block.input_shapes = Vec::new();
        block.output_shapes = Vec::new();
    }
    for edge in &mut edges {
        edge.shape = None;
    }

    // ── Cycle detection ──────────────────────────────────────────────────
    let order = match topo_sort_ids(&blocks, &edges) {
        Some(order) => order,
        None => {
            return ShapeResult {
                graph: Graph {
                    blocks,
                    edges,
                    groups: graph.groups.clone(),
                },
                errors: vec![ShapeError {
                    block_id: String::new(),
                    message: "Cycle detected in graph".to_string(),
                }],
            };
        }
    };

    // Build block id → index map
    let block_index: HashMap<String, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.clone(), i))
        .collect();

    // Build incoming edge map:  to_block_id → Vec<(from_block_id, edge_index)>
    let mut incoming_edges: HashMap<String, Vec<(String, usize)>> = HashMap::new();
    for b in &blocks {
        incoming_edges.insert(b.id.clone(), Vec::new());
    }
    for (i, e) in edges.iter().enumerate() {
        if let Some(incoming) = incoming_edges.get_mut(&e.to) {
            incoming.push((e.from.clone(), i));
        }
    }

    // Build outgoing edge map:  from_block_id → Vec<edge_index>
    let mut outgoing_edges: HashMap<String, Vec<usize>> = HashMap::new();
    for b in &blocks {
        outgoing_edges.insert(b.id.clone(), Vec::new());
    }
    for (i, e) in edges.iter().enumerate() {
        if let Some(outgoing) = outgoing_edges.get_mut(&e.from) {
            outgoing.push(i);
        }
    }

    // ── Walk blocks in topological order ───────────────────────────────
    for block_id in &order {
        let block_idx = block_index[block_id];

        // Collect input shapes from incoming edges
        let incoming = incoming_edges.get(block_id).cloned().unwrap_or_default();
        let mut input_shapes: Vec<Shape> = Vec::with_capacity(incoming.len());
        for (from_id, edge_idx) in &incoming {
            // Prefer edge.shape if already set; fall back to source block's
            // first output shape; default to empty.
            let shape = edges[*edge_idx]
                .shape
                .clone()
                .or_else(|| {
                    block_index
                        .get(from_id)
                        .and_then(|src_idx| blocks[*src_idx].output_shapes.first().cloned())
                })
                .unwrap_or_default();
            input_shapes.push(shape);
        }
        blocks[block_idx].input_shapes = input_shapes.clone();

        // Look up block definition and infer output shapes
        let def = registry.get(&blocks[block_idx].block_type);
        let output_shapes: Vec<Shape>;

        match def {
            Some(block_def) => {
                match block_def.infer_shape(&blocks[block_idx].input_shapes, &blocks[block_idx].params)
                {
                    Ok(shapes) => {
                        output_shapes = shapes;
                    }
                    Err(msg) => {
                        errors.push(ShapeError {
                            block_id: block_id.clone(),
                            message: msg,
                        });
                        output_shapes = Vec::new();
                    }
                }
            }
            None => {
                errors.push(ShapeError {
                    block_id: block_id.clone(),
                    message: format!("Unknown block type: \"{}\"", blocks[block_idx].block_type),
                });
                output_shapes = Vec::new();
            }
        }

        blocks[block_idx].output_shapes = output_shapes.clone();

        // Copy visual config from BlockDef (default to true)
        blocks[block_idx].show_depth = Some(def.map_or(true, |d| d.show_depth()));

        // Compute parameter count
        if let Some(block_def) = def {
            blocks[block_idx].param_count =
                block_def.param_count(&blocks[block_idx].input_shapes, &blocks[block_idx].params);
        }

        // Propagate first output shape to all outgoing edges
        let out_idxs = outgoing_edges.get(block_id).cloned().unwrap_or_default();
        for idx in out_idxs {
            edges[idx].shape = Some(output_shapes.first().cloned().unwrap_or_default());
        }
    }

    ShapeResult {
        graph: Graph {
            blocks,
            edges,
            groups: graph.groups.clone(),
        },
        errors,
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::graph::{Edge, Graph};
    use crate::ast::nodes::{ParamValue, ShapeVal, SourceLoc};
    use crate::block::types::{BlockDef, ParamSpec};

    // ------------------------------------------------------------------
    // Helper: dummy source location
    // ------------------------------------------------------------------
    fn dummy_loc() -> SourceLoc {
        SourceLoc {
            line: 0,
            col: 0,
            offset: 0,
        }
    }

    // ------------------------------------------------------------------
    // Helper: make a ParamValue::Number
    // ------------------------------------------------------------------
    fn p_num(value: f64) -> ParamValue {
        ParamValue::Number(Box::new(crate::ast::nodes::NumberVal {
            kind: "number".to_string(),
            value,
            loc: dummy_loc(),
        }))
    }

    // ------------------------------------------------------------------
    // Helper: make a ParamValue::Shape from dims
    // ------------------------------------------------------------------
    fn p_shape(dims: Vec<usize>) -> ParamValue {
        ParamValue::Shape(Box::new(ShapeVal {
            kind: "shape".to_string(),
            dims,
            loc: dummy_loc(),
        }))
    }

    // ------------------------------------------------------------------
    // Helper: make a block with no params
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Helper: make a block with params
    // ------------------------------------------------------------------
    fn make_block_with_params(id: &str, block_type: &str, params: HashMap<String, ParamValue>) -> Block {
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

    // ------------------------------------------------------------------
    // Helper: make an edge
    // ------------------------------------------------------------------
    fn make_edge(from: &str, to: &str) -> Edge {
        Edge {
            from: from.to_string(),
            to: to.to_string(),
            tensor_name: None,
            shape: None,
        }
    }

    // ==============================================================
    // Mock BlockDef implementations
    // ==============================================================

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

        fn param_count(&self, _inputs: &[Shape], _params: &HashMap<String, ParamValue>) -> Option<usize> {
            Some(0)
        }
    }

    /// Conv2d block: [N, C, H, W] → [N, out_channels, (H - kH + 1), (W - kW + 1)]
    struct MockConv2d;

    impl BlockDef for MockConv2d {
        fn name(&self) -> &str {
            "Conv2d"
        }

        fn params(&self) -> &[ParamSpec] {
            &[]
        }

        fn infer_shape(
            &self,
            inputs: &[Shape],
            params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<Shape>, String> {
            let input = inputs
                .first()
                .ok_or_else(|| "Conv2d requires an input shape".to_string())?;
            if input.len() != 4 {
                return Err("Conv2d expects 4D input [N, C, H, W]".to_string());
            }
            let n = input[0];
            let _c = input[1];
            let h = input[2];
            let w = input[3];

            let out_channels = params
                .get("out_channels")
                .and_then(|v| {
                    if let ParamValue::Number(n) = v {
                        Some(n.value as usize)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| "Conv2d missing out_channels".to_string())?;

            // Support kernel as shape [kH, kW] or as a single number
            let (kh, kw) = params
                .get("kernel")
                .map(|v| match v {
                    ParamValue::Shape(s) => {
                        let dims = &s.dims;
                        if dims.len() == 2 {
                            (dims[0], dims[1])
                        } else {
                            (dims[0], dims[0])
                        }
                    }
                    ParamValue::Number(n) => {
                        let k = n.value as usize;
                        (k, k)
                    }
                    _ => (3, 3),
                })
                .unwrap_or((3, 3));

            let stride = params
                .get("stride")
                .and_then(|v| {
                    if let ParamValue::Number(n) = v {
                        Some(n.value as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(1);

            let padding = params
                .get("padding")
                .and_then(|v| {
                    if let ParamValue::Number(n) = v {
                        Some(n.value as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            let out_h = (h + 2 * padding).saturating_sub(kh) / stride + 1;
            let out_w = (w + 2 * padding).saturating_sub(kw) / stride + 1;

            Ok(vec![vec![n, out_channels, out_h, out_w]])
        }

        fn param_count(
            &self,
            inputs: &[Shape],
            params: &HashMap<String, ParamValue>,
        ) -> Option<usize> {
            let in_channels = inputs.first().and_then(|s| s.get(1)).copied().unwrap_or(0);
            let out_channels = params
                .get("out_channels")
                .and_then(|v| {
                    if let ParamValue::Number(n) = v {
                        Some(n.value as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let (kh, kw) = params
                .get("kernel")
                .map(|v| match v {
                    ParamValue::Shape(s) => {
                        let dims = &s.dims;
                        if dims.len() >= 2 {
                            (dims[0], dims[1])
                        } else {
                            (dims[0], dims[0])
                        }
                    }
                    ParamValue::Number(n) => {
                        let k = n.value as usize;
                        (k, k)
                    }
                    _ => (3, 3),
                })
                .unwrap_or((3, 3));
            Some(in_channels * out_channels * kh * kw + out_channels) // weight + bias
        }
    }

    /// ReLU block: passthrough (same shape as input), show_depth = false
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

        fn show_depth(&self) -> bool {
            false
        }
    }

    /// AvgPool block: [N, C, H, W] → [N, C, (H / k), (W / k)]
    struct MockAvgPool;

    impl BlockDef for MockAvgPool {
        fn name(&self) -> &str {
            "AvgPool"
        }

        fn params(&self) -> &[ParamSpec] {
            &[]
        }

        fn infer_shape(
            &self,
            inputs: &[Shape],
            params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<Shape>, String> {
            let input = inputs
                .first()
                .ok_or_else(|| "AvgPool requires an input shape".to_string())?;
            if input.len() != 4 {
                return Err("AvgPool expects 4D input [N, C, H, W]".to_string());
            }
            let n = input[0];
            let c = input[1];
            let h = input[2];
            let w = input[3];

            let kernel = params
                .get("kernel")
                .and_then(|v| {
                    if let ParamValue::Number(n) = v {
                        Some(n.value as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(2);

            let stride = params
                .get("stride")
                .and_then(|v| {
                    if let ParamValue::Number(n) = v {
                        Some(n.value as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(kernel);

            let out_h = (h.saturating_sub(kernel)) / stride + 1;
            let out_w = (w.saturating_sub(kernel)) / stride + 1;

            Ok(vec![vec![n, c, out_h, out_w]])
        }
    }

    /// Output block: passthrough, show_depth = false
    struct MockOutput;

    impl BlockDef for MockOutput {
        fn name(&self) -> &str {
            "Output"
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

        fn show_depth(&self) -> bool {
            false
        }
    }

    /// LSTM block: multi-output. Input [batch, seq, input_size] → output [[batch, seq, hidden], [batch, hidden]]
    struct MockLSTM;

    impl BlockDef for MockLSTM {
        fn name(&self) -> &str {
            "LSTM"
        }

        fn params(&self) -> &[ParamSpec] {
            &[]
        }

        fn infer_shape(
            &self,
            inputs: &[Shape],
            params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<Shape>, String> {
            let input = inputs
                .first()
                .ok_or_else(|| "LSTM requires an input shape".to_string())?;
            if input.len() != 3 {
                return Err("LSTM expects 3D input [batch, seq, input_size]".to_string());
            }
            let batch = input[0];
            let seq = input[1];
            let hidden = params
                .get("hidden_size")
                .and_then(|v| {
                    if let ParamValue::Number(n) = v {
                        Some(n.value as usize)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| "LSTM missing hidden_size".to_string())?;

            Ok(vec![
                vec![batch, seq, hidden], // all hidden states (seq_len)
                vec![batch, hidden],      // final hidden state
            ])
        }
    }

    /// ErrorBlock: always returns Err in infer_shape
    struct MockErrorBlock;

    impl BlockDef for MockErrorBlock {
        fn name(&self) -> &str {
            "ErrorBlock"
        }

        fn params(&self) -> &[ParamSpec] {
            &[]
        }

        fn infer_shape(
            &self,
            _inputs: &[Shape],
            _params: &HashMap<String, ParamValue>,
        ) -> Result<Vec<Shape>, String> {
            Err("Something went wrong in shape inference".to_string())
        }
    }

    /// Identity block: passthrough
    struct MockIdentity;

    impl BlockDef for MockIdentity {
        fn name(&self) -> &str {
            "Identity"
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

        fn show_depth(&self) -> bool {
            false
        }
    }

    /// Merge block (elementwise add): takes 2 inputs of the same shape, passes through
    struct MockMerge;

    impl BlockDef for MockMerge {
        fn name(&self) -> &str {
            "Merge"
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
                return Err("Merge requires at least one input".to_string());
            }
            // Elementwise merge: all inputs should have the same shape
            let first = &inputs[0];
            for (i, input) in inputs.iter().enumerate().skip(1) {
                if input != first {
                    return Err(format!(
                        "Merge shape mismatch: input 0 = {:?}, input {} = {:?}",
                        first, i, input
                    ));
                }
            }
            Ok(vec![first.clone()])
        }

        fn show_depth(&self) -> bool {
            false
        }
    }

    // ==============================================================
    // Tests
    // ==============================================================

    // ── 1. Simple Input → Conv2d → ReLU → AvgPool → Output ────────────────
    #[test]
    fn test_simple_chain() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockInput));
        registry.register(Box::new(MockConv2d));
        registry.register(Box::new(MockReLU));
        registry.register(Box::new(MockAvgPool));
        registry.register(Box::new(MockOutput));

        let mut params = HashMap::new();
        params.insert("dims".to_string(), p_shape(vec![1, 3, 32, 32]));
        let input = make_block_with_params("input", "Input", params);

        let mut conv_params = HashMap::new();
        conv_params.insert("out_channels".to_string(), p_num(64.0));
        conv_params.insert("kernel".to_string(), p_shape(vec![3, 3]));
        let conv = make_block_with_params("conv", "Conv2d", conv_params);

        let relu = make_block("relu", "ReLU");
        let pool = make_block_with_params(
            "pool",
            "AvgPool",
            HashMap::from([("kernel".to_string(), p_num(2.0))]),
        );
        let output = make_block("output", "Output");

        let graph = Graph {
            blocks: vec![input, conv, relu, pool, output],
            edges: vec![
                make_edge("input", "conv"),
                make_edge("conv", "relu"),
                make_edge("relu", "pool"),
                make_edge("pool", "output"),
            ],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);

        // Check shapes propagate
        let blocks = &result.graph.blocks;
        let input_b = blocks.iter().find(|b| b.id == "input").unwrap();
        assert_eq!(input_b.output_shapes, vec![vec![1, 3, 32, 32]]);

        let conv_b = blocks.iter().find(|b| b.id == "conv").unwrap();
        // [1, 3, 32, 32] → [1, 64, 30, 30] (kernel 3, stride 1, padding 0)
        assert_eq!(conv_b.input_shapes, vec![vec![1, 3, 32, 32]]);
        assert_eq!(conv_b.output_shapes, vec![vec![1, 64, 30, 30]]);
        assert_eq!(conv_b.param_count, Some(3 * 64 * 3 * 3 + 64)); // 1728 + 64 = 1792

        let relu_b = blocks.iter().find(|b| b.id == "relu").unwrap();
        assert_eq!(relu_b.input_shapes, vec![vec![1, 64, 30, 30]]);
        assert_eq!(relu_b.output_shapes, vec![vec![1, 64, 30, 30]]);
        assert_eq!(relu_b.show_depth, Some(false));

        let pool_b = blocks.iter().find(|b| b.id == "pool").unwrap();
        // [1, 64, 30, 30] → [1, 64, 15, 15] (kernel 2, stride 2)
        assert_eq!(pool_b.input_shapes, vec![vec![1, 64, 30, 30]]);
        assert_eq!(pool_b.output_shapes, vec![vec![1, 64, 15, 15]]);

        let output_b = blocks.iter().find(|b| b.id == "output").unwrap();
        assert_eq!(output_b.input_shapes, vec![vec![1, 64, 15, 15]]);
        assert_eq!(output_b.output_shapes, vec![vec![1, 64, 15, 15]]);
        assert_eq!(output_b.show_depth, Some(false));

        // Check edge shapes
        let edges = &result.graph.edges;
        for e in edges {
            assert!(e.shape.is_some(), "edge {:?}->{:?} has no shape", e.from, e.to);
        }
    }

    // ── 2. Multi-output block (LSTM) ──────────────────────────────────────
    #[test]
    fn test_multi_output_lstm() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockInput));
        registry.register(Box::new(MockLSTM));
        registry.register(Box::new(MockIdentity));
        registry.register(Box::new(MockIdentity));

        let input = make_block_with_params(
            "input",
            "Input",
            HashMap::from([("dims".to_string(), p_shape(vec![4, 10, 32]))]),
        );

        let lstm = make_block_with_params(
            "lstm",
            "LSTM",
            HashMap::from([("hidden_size".to_string(), p_num(64.0))]),
        );

        let out1 = make_block("out1", "Identity");
        let out2 = make_block("out2", "Identity");

        // LSTM has 2 outputs: out1 connects to first output, out2 connects to second
        let graph = Graph {
            blocks: vec![input, lstm, out1, out2],
            edges: vec![
                make_edge("input", "lstm"),
                make_edge("lstm", "out1"),
                make_edge("lstm", "out2"),
            ],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);

        let lstm_b = result
            .graph
            .blocks
            .iter()
            .find(|b| b.id == "lstm")
            .unwrap();

        // LSTM outputs [batch, seq, hidden] = [4, 10, 64] and [batch, hidden] = [4, 64]
        assert_eq!(lstm_b.output_shapes.len(), 2);
        assert_eq!(lstm_b.output_shapes[0], vec![4, 10, 64]);
        assert_eq!(lstm_b.output_shapes[1], vec![4, 64]);

        // Both outgoing edges should get the FIRST output shape (this matches TS behavior)
        let edges = &result.graph.edges;
        for e in edges {
            if e.from == "lstm" {
                // TS propagates outputShapes[0] to all edges, so both get [4, 10, 64]
                assert_eq!(e.shape, Some(vec![4, 10, 64]));
            }
        }
    }

    // ── 3. Unknown block type error ───────────────────────────────────────
    #[test]
    fn test_unknown_block_type() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockInput));
        registry.register(Box::new(MockOutput));
        let input = make_block("input", "Input");
        let unknown = make_block("unknown", "NonExistentBlock");
        let output = make_block("output", "Output");

        let graph = Graph {
            blocks: vec![input, unknown, output],
            edges: vec![
                make_edge("input", "unknown"),
                make_edge("unknown", "output"),
            ],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        // Should have one error for the unknown block
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].block_id, "unknown");
        assert!(result.errors[0].message.contains("NonExistentBlock"));

        // Unknown block should have empty output shapes
        let unk = result
            .graph
            .blocks
            .iter()
            .find(|b| b.id == "unknown")
            .unwrap();
        assert!(unk.output_shapes.is_empty());
    }

    // ── 4. Cycle detection ────────────────────────────────────────────────
    #[test]
    fn test_cycle_detection() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockIdentity));

        let a = make_block("a", "Identity");
        let b = make_block("b", "Identity");

        // a → b → a (cycle)
        let graph = Graph {
            blocks: vec![a, b],
            edges: vec![make_edge("a", "b"), make_edge("b", "a")],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].block_id, "");
        assert!(result.errors[0].message.contains("Cycle"));
    }

    // ── 5. Missing required params (infer_shape returns Err) ──────────────
    #[test]
    fn test_infer_shape_error() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockConv2d));

        // Conv2d without required params
        let conv = make_block("conv", "Conv2d");
        let output = make_block("output", "Output");

        let graph = Graph {
            blocks: vec![conv, output],
            edges: vec![make_edge("conv", "output")],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        // Conv2d will fail because no input shape
        // Actually, Conv2d will get an empty input shape (since there's no incoming edge providing it)
        // and also missing out_channels param
        assert!(!result.errors.is_empty());
        let conv_err = result.errors.iter().find(|e| e.block_id == "conv");
        assert!(conv_err.is_some());
    }

    // ── 5b. infer_shape returns explicit Err via ErrorBlock ────────────────
    #[test]
    fn test_error_block_returns_err() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockErrorBlock));
        registry.register(Box::new(MockOutput));

        let err_block = make_block("err", "ErrorBlock");
        let output = make_block("output", "Output");

        let graph = Graph {
            blocks: vec![err_block, output],
            edges: vec![make_edge("err", "output")],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].block_id, "err");
        assert!(result.errors[0].message.contains("Something went wrong"));

        // ErrorBlock should have empty output shapes
        let err_b = result.graph.blocks.iter().find(|b| b.id == "err").unwrap();
        assert!(err_b.output_shapes.is_empty());
    }

    // ── 6. LeNet5 full model shape trace ──────────────────────────────────
    #[test]
    fn test_lenet5_shape_trace() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockInput));
        registry.register(Box::new(MockConv2d));
        registry.register(Box::new(MockReLU));
        registry.register(Box::new(MockAvgPool));
        registry.register(Box::new(MockOutput));

        // LeNet5: Input(1,28,28) → Conv2d(6,5) → ReLU → AvgPool(2)
        //         → Conv2d(16,5) → ReLU → AvgPool(2) → Output
        let input = make_block_with_params(
            "input",
            "Input",
            HashMap::from([("dims".to_string(), p_shape(vec![1, 1, 28, 28]))]),
        );

        let conv1 = make_block_with_params(
            "conv1",
            "Conv2d",
            HashMap::from([
                ("out_channels".to_string(), p_num(6.0)),
                ("kernel".to_string(), p_shape(vec![5, 5])),
            ]),
        );
        let relu1 = make_block("relu1", "ReLU");
        let pool1 = make_block_with_params(
            "pool1",
            "AvgPool",
            HashMap::from([("kernel".to_string(), p_num(2.0))]),
        );

        let conv2 = make_block_with_params(
            "conv2",
            "Conv2d",
            HashMap::from([
                ("out_channels".to_string(), p_num(16.0)),
                ("kernel".to_string(), p_shape(vec![5, 5])),
            ]),
        );
        let relu2 = make_block("relu2", "ReLU");
        let pool2 = make_block_with_params(
            "pool2",
            "AvgPool",
            HashMap::from([("kernel".to_string(), p_num(2.0))]),
        );
        let output = make_block("output", "Output");

        let graph = Graph {
            blocks: vec![
                input, conv1, relu1, pool1, conv2, relu2, pool2, output,
            ],
            edges: vec![
                make_edge("input", "conv1"),
                make_edge("conv1", "relu1"),
                make_edge("relu1", "pool1"),
                make_edge("pool1", "conv2"),
                make_edge("conv2", "relu2"),
                make_edge("relu2", "pool2"),
                make_edge("pool2", "output"),
            ],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);

        let bmap: HashMap<&str, &Block> = result
            .graph
            .blocks
            .iter()
            .map(|b| (b.id.as_str(), b))
            .collect();

        // Input: [1, 1, 28, 28]
        assert_eq!(bmap["input"].output_shapes, vec![vec![1, 1, 28, 28]]);

        // Conv1: [1, 1, 28, 28] → [1, 6, 24, 24] (kernel 5)
        assert_eq!(bmap["conv1"].output_shapes, vec![vec![1, 6, 24, 24]]);

        // ReLU1: passthrough
        assert_eq!(bmap["relu1"].output_shapes, vec![vec![1, 6, 24, 24]]);

        // Pool1: [1, 6, 24, 24] → [1, 6, 12, 12] (kernel 2, stride 2)
        // (24 - 2) / 2 + 1 = 11 + 1 = 12
        assert_eq!(bmap["pool1"].output_shapes, vec![vec![1, 6, 12, 12]]);

        // Conv2: [1, 6, 12, 12] → [1, 16, 8, 8] (kernel 5)
        // (12 - 5) / 1 + 1 = 8
        assert_eq!(bmap["conv2"].output_shapes, vec![vec![1, 16, 8, 8]]);

        // ReLU2: passthrough
        assert_eq!(bmap["relu2"].output_shapes, vec![vec![1, 16, 8, 8]]);

        // Pool2: [1, 16, 8, 8] → [1, 16, 4, 4] (kernel 2, stride 2)
        // (8 - 2) / 2 + 1 = 3 + 1 = 4
        assert_eq!(bmap["pool2"].output_shapes, vec![vec![1, 16, 4, 4]]);

        // Output: passthrough
        assert_eq!(bmap["output"].output_shapes, vec![vec![1, 16, 4, 4]]);
    }

    // ── 7. Input → Output passthrough ─────────────────────────────────────
    #[test]
    fn test_input_output_passthrough() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockInput));
        registry.register(Box::new(MockOutput));

        let input = make_block_with_params(
            "input",
            "Input",
            HashMap::from([("dims".to_string(), p_shape(vec![1, 3, 224, 224]))]),
        );
        let output = make_block("output", "Output");

        let graph = Graph {
            blocks: vec![input, output],
            edges: vec![make_edge("input", "output")],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);

        let input_b = result
            .graph
            .blocks
            .iter()
            .find(|b| b.id == "input")
            .unwrap();
        assert_eq!(input_b.output_shapes, vec![vec![1, 3, 224, 224]]);

        let output_b = result
            .graph
            .blocks
            .iter()
            .find(|b| b.id == "output")
            .unwrap();
        assert_eq!(output_b.output_shapes, vec![vec![1, 3, 224, 224]]);

        let edge = &result.graph.edges[0];
        assert_eq!(edge.shape, Some(vec![1, 3, 224, 224]));
    }

    // ── 8. Elementwise merge shape compatibility ──────────────────────────
    #[test]
    fn test_merge_compatibility() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockInput));
        registry.register(Box::new(MockMerge));
        registry.register(Box::new(MockOutput));

        let input_a = make_block_with_params(
            "input_a",
            "Input",
            HashMap::from([("dims".to_string(), p_shape(vec![1, 3, 32, 32]))]),
        );
        let input_b = make_block_with_params(
            "input_b",
            "Input",
            HashMap::from([("dims".to_string(), p_shape(vec![1, 3, 32, 32]))]),
        );
        let merge = make_block("merge", "Merge");
        let output = make_block("output", "Output");

        let graph = Graph {
            blocks: vec![input_a, input_b, merge, output],
            edges: vec![
                make_edge("input_a", "merge"),
                make_edge("input_b", "merge"),
                make_edge("merge", "output"),
            ],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);

        let merge_b = result
            .graph
            .blocks
            .iter()
            .find(|b| b.id == "merge")
            .unwrap();
        assert_eq!(merge_b.input_shapes, vec![vec![1, 3, 32, 32], vec![1, 3, 32, 32]]);
        assert_eq!(merge_b.output_shapes, vec![vec![1, 3, 32, 32]]);
    }

    // ── 9. Merge shape mismatch should error ──────────────────────────────
    #[test]
    fn test_merge_shape_mismatch() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockInput));
        registry.register(Box::new(MockMerge));
        registry.register(Box::new(MockOutput));

        let input_a = make_block_with_params(
            "input_a",
            "Input",
            HashMap::from([("dims".to_string(), p_shape(vec![1, 3, 32, 32]))]),
        );
        let input_b = make_block_with_params(
            "input_b",
            "Input",
            HashMap::from([("dims".to_string(), p_shape(vec![1, 3, 64, 64]))]),
        );
        let merge = make_block("merge", "Merge");
        let output = make_block("output", "Output");

        let graph = Graph {
            blocks: vec![input_a, input_b, merge, output],
            edges: vec![
                make_edge("input_a", "merge"),
                make_edge("input_b", "merge"),
                make_edge("merge", "output"),
            ],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        // Merge should report a shape mismatch error
        let merge_err = result.errors.iter().find(|e| e.block_id == "merge");
        assert!(merge_err.is_some(), "expected merge shape mismatch error");
        assert!(merge_err.unwrap().message.contains("shape mismatch"));
    }

    // ── 10. Original graph is not mutated ─────────────────────────────────
    #[test]
    fn test_original_graph_not_mutated() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockInput));
        registry.register(Box::new(MockOutput));

        let input = make_block_with_params(
            "input",
            "Input",
            HashMap::from([("dims".to_string(), p_shape(vec![1, 3, 32, 32]))]),
        );
        let output = make_block("output", "Output");

        let original_graph = Graph {
            blocks: vec![input, output],
            edges: vec![make_edge("input", "output")],
            groups: vec![],
        };

        let _result = infer_shapes(&original_graph, &registry);

        // Original graph should still have empty shapes
        for block in &original_graph.blocks {
            assert!(block.input_shapes.is_empty());
            assert!(block.output_shapes.is_empty());
        }
        for edge in &original_graph.edges {
            assert!(edge.shape.is_none());
        }
    }

    // ── 11. show_depth and param_count for registered vs unregistered blocks
    #[test]
    fn test_show_depth_and_param_count() {
        let mut registry = BlockRegistry::new();
        registry.register(Box::new(MockInput));
        registry.register(Box::new(MockConv2d));
        registry.register(Box::new(MockReLU)); // show_depth = false

        let input = make_block_with_params(
            "input",
            "Input",
            HashMap::from([("dims".to_string(), p_shape(vec![1, 3, 32, 32]))]),
        );
        let conv = make_block_with_params(
            "conv",
            "Conv2d",
            HashMap::from([
                ("out_channels".to_string(), p_num(64.0)),
                ("kernel".to_string(), p_shape(vec![3, 3])),
            ]),
        );
        let relu = make_block("relu", "ReLU");
        let unknown = make_block("unk", "UnknownBlock");

        let graph = Graph {
            blocks: vec![input, conv, relu, unknown],
            edges: vec![
                make_edge("input", "conv"),
                make_edge("conv", "relu"),
                make_edge("relu", "unknown"),
            ],
            groups: vec![],
        };

        let result = infer_shapes(&graph, &registry);

        // Known blocks should have show_depth = Some(value)
        let input_b = result.graph.blocks.iter().find(|b| b.id == "input").unwrap();
        assert_eq!(input_b.show_depth, Some(true));
        assert_eq!(input_b.param_count, Some(0));

        let conv_b = result.graph.blocks.iter().find(|b| b.id == "conv").unwrap();
        assert_eq!(conv_b.show_depth, Some(true));
        assert_eq!(conv_b.param_count, Some(3 * 64 * 3 * 3 + 64));

        let relu_b = result.graph.blocks.iter().find(|b| b.id == "relu").unwrap();
        assert_eq!(relu_b.show_depth, Some(false)); // ReLU overrides show_depth

        // Unknown block: show_depth defaults to true, param_count is None
        let unk_b = result.graph.blocks.iter().find(|b| b.id == "unk").unwrap();
        assert_eq!(unk_b.show_depth, Some(true));
        assert_eq!(unk_b.param_count, None);

        // Should have one error for unknown block
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].block_id, "unk");
    }
}
