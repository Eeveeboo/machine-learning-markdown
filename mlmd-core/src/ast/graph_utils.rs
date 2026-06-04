use std::collections::{HashMap, VecDeque};

use crate::ast::graph::{Block, Edge, Graph};

// ---------------------------------------------------------------------------
// NodeInfo — adjacency information for a single block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub block: Block,
    /// Block ids that feed into this block
    pub inputs: Vec<String>,
    /// Block ids this block feeds into
    pub outputs: Vec<String>,
}

// ---------------------------------------------------------------------------
// build_adjacency — build a map from block id to NodeInfo
// ---------------------------------------------------------------------------

pub fn build_adjacency(graph: &Graph) -> HashMap<String, NodeInfo> {
    let mut map: HashMap<String, NodeInfo> = HashMap::new();
    for b in graph.blocks.iter() {
        map.insert(
            b.id.clone(),
            NodeInfo {
                block: b.clone(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
        );
    }
    for e in graph.edges.iter() {
        if let Some(from) = map.get_mut(&e.from) {
            from.outputs.push(e.to.clone());
        }
        if let Some(to) = map.get_mut(&e.to) {
            to.inputs.push(e.from.clone());
        }
    }
    map
}

// ---------------------------------------------------------------------------
// DFS-based topological sort (assumes DAG, no cycle detection)
// Returns blocks in topological order (inputs before outputs).
// ---------------------------------------------------------------------------

pub fn topo_sort(graph: &Graph) -> Vec<Block> {
    let adj = build_adjacency(graph);

    let mut visited: HashMap<String, bool> = HashMap::new();
    for b in &graph.blocks {
        visited.insert(b.id.clone(), false);
    }

    fn visit(
        id: &str,
        adj: &HashMap<String, NodeInfo>,
        visited: &mut HashMap<String, bool>,
        result: &mut Vec<Block>,
    ) {
        if visited.get(id) == Some(&true) {
            return;
        }
        visited.insert(id.to_string(), true);
        if let Some(info) = adj.get(id) {
            for inp in &info.inputs {
                visit(inp, adj, visited, result);
            }
            result.push(info.block.clone());
        }
    }

    let mut result: Vec<Block> = Vec::new();
    for b in &graph.blocks {
        visit(&b.id, &adj, &mut visited, &mut result);
    }
    result
}

// ---------------------------------------------------------------------------
// Kahn's algorithm topological sort with cycle detection
// Returns sorted block IDs, or None if a cycle is detected.
// ---------------------------------------------------------------------------

pub fn topo_sort_ids(blocks: &[Block], edges: &[Edge]) -> Option<Vec<String>> {
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut adj_out: HashMap<String, Vec<String>> = HashMap::new();

    for b in blocks {
        in_degree.insert(b.id.clone(), 0);
        adj_out.insert(b.id.clone(), Vec::new());
    }

    for e in edges {
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

    let mut result: Vec<String> = Vec::new();
    while let Some(id) = queue.pop_front() {
        result.push(id.clone());
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

    if result.len() == blocks.len() {
        Some(result)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::graph::*;
    use crate::ast::nodes::SourceLoc;

    fn dummy_loc() -> SourceLoc {
        SourceLoc {
            line: 0,
            col: 0,
            offset: 0,
        }
    }

    fn make_block(id: &str) -> Block {
        Block {
            id: id.to_string(),
            block_type: "Dummy".to_string(),
            params: std::collections::HashMap::new(),
            input_shapes: vec![],
            output_shapes: vec![],
            param_count: None,
            show_depth: None,
            loc: dummy_loc(),
        }
    }

    #[test]
    fn test_topo_sort_simple() {
        let b0 = make_block("b0");
        let b1 = make_block("b1");
        let b2 = make_block("b2");

        // b0 -> b1 -> b2
        let graph = Graph {
            blocks: vec![b0, b1.clone(), b2.clone()],
            edges: vec![
                Edge {
                    from: "b0".to_string(),
                    to: "b1".to_string(),
                    tensor_name: None,
                    shape: None,
                },
                Edge {
                    from: "b1".to_string(),
                    to: "b2".to_string(),
                    tensor_name: None,
                    shape: None,
                },
            ],
            groups: vec![],
        };

        let sorted = topo_sort(&graph);
        let ids: Vec<&str> = sorted.iter().map(|b| b.id.as_str()).collect();
        assert_eq!(ids, vec!["b0", "b1", "b2"]);
    }

    #[test]
    fn test_topo_sort_ids_acyclic() {
        let blocks = vec![make_block("b0"), make_block("b1"), make_block("b2")];
        let edges = vec![
            Edge {
                from: "b0".to_string(),
                to: "b1".to_string(),
                tensor_name: None,
                shape: None,
            },
            Edge {
                from: "b0".to_string(),
                to: "b2".to_string(),
                tensor_name: None,
                shape: None,
            },
        ];

        let result = topo_sort_ids(&blocks, &edges);
        assert!(result.is_some());
        let ids = result.unwrap();
        assert_eq!(ids[0], "b0");
    }

    #[test]
    fn test_topo_sort_ids_cycle() {
        let blocks = vec![make_block("b0"), make_block("b1")];
        let edges = vec![
            Edge {
                from: "b0".to_string(),
                to: "b1".to_string(),
                tensor_name: None,
                shape: None,
            },
            Edge {
                from: "b1".to_string(),
                to: "b0".to_string(),
                tensor_name: None,
                shape: None,
            },
        ];

        let result = topo_sort_ids(&blocks, &edges);
        assert!(result.is_none());
    }
}
