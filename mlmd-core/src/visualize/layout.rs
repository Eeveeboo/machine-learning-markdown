use std::collections::HashMap;

use crate::ast::graph::{Block, Edge, Graph};
use crate::ast::graph_utils::topo_sort_ids;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MIN_BLOCK_W: f64 = 60.0;
const MAX_BLOCK_W: f64 = 300.0;
const MIN_BLOCK_H: f64 = 28.0;
const MAX_BLOCK_H: f64 = 200.0;
const V_GAP: f64 = 24.0;
const H_GAP: f64 = 10.0;
const LABEL_H: f64 = 12.0;
const LABEL_CHAR_W: f64 = 5.0;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A 2D point.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// A laid-out node (block with computed position and dimensions).
#[derive(Debug, Clone)]
pub struct LayoutNode {
    pub block: Block,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// A laid-out edge with routing points.
#[derive(Debug, Clone)]
pub struct LayoutEdge {
    pub from: LayoutNode,
    pub to: LayoutNode,
    pub points: Vec<Point>,
    pub is_long_range: bool,
    pub label: Option<String>,
    /// Rightwards offset for the bezier curve (long-range edges only).
    pub curve_offset: Option<f64>,
    pub route_side: Option<String>,
    pub exit_port: Option<Point>,
    pub entry_port: Option<Point>,
    pub label_position: Option<Point>,
}

/// A laid-out group with bounding box.
#[derive(Debug, Clone)]
pub struct GroupLayout {
    pub path: Vec<String>,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// The complete layout result for a graph.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    pub nodes: Vec<LayoutNode>,
    pub edges: Vec<LayoutEdge>,
    pub groups: Vec<GroupLayout>,
    pub width: f64,
    pub height: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute block width from the spatial width (last dimension) of the output tensor.
fn compute_block_width(block: &Block) -> f64 {
    let shape = match block.output_shapes.first() {
        Some(s) if !s.is_empty() => s,
        _ => return 0.0,
    };
    let last = *shape.last().unwrap();
    let w = if shape.len() >= 2 {
        last as f64 * last as f64
    } else {
        last as f64 * last as f64
    };
    w.sqrt() * 8.0
}

/// Compute block height from channel depth (first dimension) of the output tensor.
/// Returns 0 for 1D shapes or when `show_depth` is false.
fn compute_block_height(block: &Block) -> f64 {
    if block.show_depth == Some(false) {
        return 0.0;
    }
    let shape = match block.output_shapes.first() {
        Some(s) if s.len() >= 3 => s,
        _ => return 0.0,
    };
    let c = shape[0] as f64;
    c.sqrt() * 8.0
}

/// Choose which side (left or right) a long-range edge should route on.
pub fn choose_side(
    from_node: &LayoutNode,
    to_node: &LayoutNode,
    all_nodes: &[LayoutNode],
    from_layer: usize,
    to_layer: usize,
    layer_map: &HashMap<String, usize>,
    total_width: f64,
) -> &'static str {
    let left_bound = from_node.x.min(to_node.x);
    let right_bound = (from_node.x + from_node.width).max(to_node.x + to_node.width);

    let mut left_cost = 0;
    let mut right_cost = 0;

    for node in all_nodes {
        let layer = match layer_map.get(&node.block.id) {
            Some(l) => *l,
            None => continue,
        };
        if layer <= from_layer || layer >= to_layer {
            continue;
        }
        let nx = node.x;
        let nx_end = node.x + node.width;

        // overlaps left gutter [0, leftBound]
        if nx < left_bound && nx_end > 0.0 {
            left_cost += 1;
        }
        // overlaps right gutter [rightBound, totalWidth]
        if nx < total_width && nx_end > right_bound {
            right_cost += 1;
        }
    }

    if left_cost < right_cost {
        return "left";
    }
    if right_cost < left_cost {
        return "right";
    }
    let from_center = from_node.x + from_node.width / 2.0;
    if from_center < total_width / 2.0 {
        "left"
    } else {
        "right"
    }
}

/// Compute a port position on a node.
fn compute_ports(
    node: &LayoutNode,
    is_long_range: bool,
    route_side: Option<&str>,
    is_entry: bool,
    index: usize,
    count: usize,
) -> Point {
    if is_long_range {
        match route_side {
            Some("left") => {
                return Point {
                    x: node.x,
                    y: node.y + node.height / 2.0,
                };
            }
            Some("right") => {
                return Point {
                    x: node.x + node.width,
                    y: node.y + node.height / 2.0,
                };
            }
            _ => {}
        }
    }
    // Distribute across edge (25%..75% of node width) when multiple edges
    let port_x = if count > 1 {
        node.x + node.width * (0.25 + 0.5 * (index as f64 / (count - 1) as f64))
    } else {
        node.x + node.width / 2.0
    };
    if is_entry {
        Point {
            x: port_x,
            y: node.y,
        }
    } else {
        Point {
            x: port_x,
            y: node.y + node.height,
        }
    }
}

// ---------------------------------------------------------------------------
// Label overlap helpers
// ---------------------------------------------------------------------------

/// A label bounding rectangle.
#[derive(Debug, Clone)]
pub struct LabelRect {
    pub edge_index: usize,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Detect overlapping label rectangles. Returns a list of pairs of indices.
pub fn detect_label_overlaps(labels: &[LabelRect]) -> Vec<(usize, usize)> {
    let mut overlaps = Vec::new();
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len() {
            let a = &labels[i];
            let b = &labels[j];
            let overlap_x = a.x < b.x + b.width && a.x + a.width > b.x;
            let overlap_y = a.y < b.y + b.height && a.y + a.height > b.y;
            if overlap_x && overlap_y {
                overlaps.push((i, j));
            }
        }
    }
    overlaps
}

/// Resolve overlapping labels by shifting them down.
pub fn resolve_overlaps(labels: &mut [LabelRect], max_iterations: usize) {
    for _iter in 0..max_iterations {
        let overlaps = detect_label_overlaps(labels);
        if overlaps.is_empty() {
            break;
        }
        for &(i, j) in &overlaps {
            let a = labels[i].y + labels[i].height;
            let b_y = labels[j].y;
            let overlap_amount = a - b_y;
            if overlap_amount > 0.0 {
                labels[j].y += overlap_amount + 4.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Group layout
// ---------------------------------------------------------------------------

struct GroupBuilder {
    path: Vec<String>,
    nodes: Vec<LayoutNode>,
}

/// Compute group bounding boxes and ensure no overlap between adjacent groups.
fn compute_group_layouts(
    graph: &Graph,
    node_map: &HashMap<String, LayoutNode>,
) -> Vec<GroupLayout> {
    let side_pad = 10.0;
    let top_pad = 28.0;
    let bot_pad = 10.0;
    let min_gap = 6.0;

    let mut builders: Vec<GroupBuilder> = Vec::new();
    for group in &graph.groups {
        let g_nodes: Vec<LayoutNode> = group
            .block_ids
            .iter()
            .filter_map(|id| node_map.get(id).cloned())
            .collect();
        if g_nodes.is_empty() {
            continue;
        }
        builders.push(GroupBuilder {
            path: group.path.clone(),
            nodes: g_nodes,
        });
    }

    // Sort by Y position of topmost node
    builders.sort_by(|a, b| {
        let a_min_y = a.nodes.iter().map(|n| n.y).fold(f64::INFINITY, f64::min);
        let b_min_y = b.nodes.iter().map(|n| n.y).fold(f64::INFINITY, f64::min);
        a_min_y
            .partial_cmp(&b_min_y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut groups: Vec<GroupLayout> = Vec::new();

    for mut b in builders {
        let min_x = b.nodes.iter().map(|n| n.x).fold(f64::INFINITY, f64::min) - side_pad;
        let min_y = b.nodes.iter().map(|n| n.y).fold(f64::INFINITY, f64::min) - top_pad;
        let max_x = b
            .nodes
            .iter()
            .map(|n| n.x + n.width)
            .fold(f64::NEG_INFINITY, f64::max)
            + side_pad;
        let max_y = b
            .nodes
            .iter()
            .map(|n| n.y + n.height)
            .fold(f64::NEG_INFINITY, f64::max)
            + bot_pad;

        let x = min_x;
        let mut y = min_y;
        let width = max_x - min_x;
        let mut height = max_y - min_y;

        // Resolve vertical overlap with previous groups
        if let Some(prev) = groups.last() {
            let prev_bottom = prev.y + prev.height;
            if y < prev_bottom + min_gap {
                let shift = prev_bottom + min_gap - y;
                y += shift;
                // Shift member nodes to keep them aligned with the group box
                for node in &mut b.nodes {
                    node.y += shift;
                    height = height.max(node.y + node.height + bot_pad - y);
                }
            }
        }

        groups.push(GroupLayout {
            path: b.path,
            x,
            y,
            width,
            height,
        });
    }

    groups
}

// ---------------------------------------------------------------------------
// Main layout function
// ---------------------------------------------------------------------------

/// Compute node positions, edge paths, and group bounding boxes for a graph.
pub fn layout(graph: &Graph) -> LayoutResult {
    let blocks = &graph.blocks;
    let edges = &graph.edges;

    if blocks.is_empty() {
        return LayoutResult {
            nodes: Vec::new(),
            edges: Vec::new(),
            groups: Vec::new(),
            width: 0.0,
            height: 0.0,
        };
    }

    // Topological sort
    let sorted = match topo_sort_ids(blocks, edges) {
        Some(s) => s,
        None => {
            // Fallback: use block order if graph has cycles
            blocks.iter().map(|b| b.id.clone()).collect()
        }
    };

    // Assign layers (longest path from sources)
    let layer_map = assign_layers(&sorted, edges);

    // Group blocks by layer
    let mut by_layer: HashMap<usize, Vec<String>> = HashMap::new();
    for id in &sorted {
        let l = *layer_map.get(id).unwrap_or(&0);
        by_layer.entry(l).or_default().push(id.clone());
    }

    let mut sorted_layers: Vec<usize> = by_layer.keys().copied().collect();
    sorted_layers.sort();

    let block_by_id: HashMap<String, Block> =
        blocks.iter().map(|b| (b.id.clone(), b.clone())).collect();

    // Pre-compute widths and heights for all blocks
    let mut width_map: HashMap<String, f64> = HashMap::new();
    let mut height_map: HashMap<String, f64> = HashMap::new();
    for b in blocks {
        width_map.insert(b.id.clone(), compute_block_width(b));
        height_map.insert(b.id.clone(), compute_block_height(b));
    }

    // Rescale all widths and heights proportionally, respecting MAX/MIN bounds
    let max_width = width_map.values().cloned().fold(0.0_f64, f64::max);
    let max_height = height_map.values().cloned().fold(0.0_f64, f64::max);
    let scale_w = if max_width > MIN_BLOCK_W {
        MAX_BLOCK_W / (max_width - MIN_BLOCK_W)
    } else {
        1.0
    };
    let scale_h = if max_height > MIN_BLOCK_H {
        MAX_BLOCK_H / (max_height - MIN_BLOCK_H)
    } else {
        1.0
    };

    for id in width_map.keys().cloned().collect::<Vec<_>>() {
        let w = width_map[&id];
        let h = height_map[&id];
        let scaled_w = (MIN_BLOCK_W + (w - MIN_BLOCK_W) * scale_w).max(MIN_BLOCK_W);
        let scaled_h = (MIN_BLOCK_H + (h - MIN_BLOCK_H) * scale_h).max(MIN_BLOCK_H);
        width_map.insert(id.clone(), scaled_w);
        height_map.insert(id.clone(), scaled_h);
    }

    // Compute max height per layer
    let mut layer_height: HashMap<usize, f64> = HashMap::new();
    for (&layer_idx, ids) in &by_layer {
        let max_h = ids
            .iter()
            .map(|id| height_map.get(id).copied().unwrap_or(0.0))
            .fold(0.0_f64, f64::max);
        layer_height.insert(layer_idx, max_h);
    }

    // Compute Y positions with variable-height stacking
    let mut layer_y: HashMap<usize, f64> = HashMap::new();
    let mut current_y = 0.0;
    for &layer_idx in &sorted_layers {
        layer_y.insert(layer_idx, current_y);
        current_y += layer_height.get(&layer_idx).copied().unwrap_or(0.0) + V_GAP;
    }

    // Find the maximum row width to center everything
    let all_row_widths: Vec<f64> = sorted_layers
        .iter()
        .map(|l| {
            let ids = &by_layer[l];
            let total_w: f64 = ids
                .iter()
                .map(|id| width_map.get(id).copied().unwrap_or(0.0))
                .sum();
            total_w + (ids.len().saturating_sub(1)) as f64 * H_GAP
        })
        .collect();
    let max_row_width = all_row_widths.iter().cloned().fold(0.0_f64, f64::max);

    // Layout: center each row horizontally
    let mut node_map: HashMap<String, LayoutNode> = HashMap::new();
    for &layer_idx in &sorted_layers {
        let ids = &by_layer[&layer_idx];
        let widths: Vec<f64> = ids
            .iter()
            .map(|id| width_map.get(id).copied().unwrap_or(0.0))
            .collect();
        let total_row_width: f64 =
            widths.iter().sum::<f64>() + (ids.len().saturating_sub(1)) as f64 * H_GAP;
        let y = layer_y.get(&layer_idx).copied().unwrap_or(0.0);
        let h = layer_height.get(&layer_idx).copied().unwrap_or(0.0);

        let mut cursor_x = (max_row_width - total_row_width) / 2.0;
        for (i, id) in ids.iter().enumerate() {
            let w = widths[i];
            let bh = height_map.get(id).copied().unwrap_or(0.0);
            // Center vertically within the layer's max height
            let by = y + (h - bh) / 2.0;
            if let Some(block) = block_by_id.get(id) {
                node_map.insert(
                    id.clone(),
                    LayoutNode {
                        block: block.clone(),
                        x: cursor_x,
                        y: by,
                        width: w,
                        height: bh,
                    },
                );
            }
            cursor_x += w + H_GAP;
        }
    }

    let layout_result_height = current_y;

    // Compute overall width
    let all_widths: Vec<f64> = sorted_layers
        .iter()
        .map(|l| {
            let ids = &by_layer[l];
            let w: f64 = ids
                .iter()
                .map(|id| width_map.get(id).copied().unwrap_or(0.0))
                .sum();
            w + (ids.len().saturating_sub(1)) as f64 * H_GAP
        })
        .collect();
    let total_width = all_widths.iter().cloned().fold(0.0_f64, f64::max);

    // Compute group bounds — may shift member nodes to resolve overlap
    let group_layouts = compute_group_layouts(graph, &node_map);

    // Pre-compute outgoing/incoming edge ordering for port distribution
    let mut out_edge_order: HashMap<String, Vec<(String, usize, usize)>> = HashMap::new(); // (target_id, index, count)
    let mut in_edge_order: HashMap<String, Vec<(String, usize, usize)>> = HashMap::new(); // (source_id, index, count)

    for e in edges {
        if !node_map.contains_key(&e.from) || !node_map.contains_key(&e.to) {
            continue;
        }
        let from_layer = layer_map.get(&e.from).copied().unwrap_or(0);
        let to_layer = layer_map.get(&e.to).copied().unwrap_or(0);
        let span = to_layer.saturating_sub(from_layer);
        if span > 2 {
            continue; // skip long-range edges
        }
        out_edge_order
            .entry(e.from.clone())
            .or_default()
            .push((e.to.clone(), 0, 0));
        in_edge_order
            .entry(e.to.clone())
            .or_default()
            .push((e.from.clone(), 0, 0));
    }

    // Sort outgoing by target x (left target → left port), assign indices
    for list in out_edge_order.values_mut() {
        list.sort_by(|(a_id, ..), (b_id, ..)| {
            let ax = node_map.get(a_id).map(|n| n.x).unwrap_or(0.0);
            let bx = node_map.get(b_id).map(|n| n.x).unwrap_or(0.0);
            ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
        });
        let len = list.len();
        for (i, item) in list.iter_mut().enumerate() {
            item.1 = i;
            item.2 = len;
        }
    }

    // Sort incoming by source x (left source → left port), assign indices
    for list in in_edge_order.values_mut() {
        list.sort_by(|(a_id, ..), (b_id, ..)| {
            let ax = node_map.get(a_id).map(|n| n.x).unwrap_or(0.0);
            let bx = node_map.get(b_id).map(|n| n.x).unwrap_or(0.0);
            ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
        });
        let len = list.len();
        for (i, item) in list.iter_mut().enumerate() {
            item.1 = i;
            item.2 = len;
        }
    }

    // Build lookup: "from__to" -> (out_index, out_count, in_index, in_count)
    #[derive(Default, Clone, Copy)]
    struct PortInfo {
        out_index: usize,
        out_count: usize,
        in_index: usize,
        in_count: usize,
    }
    let mut edge_port_info: HashMap<String, PortInfo> = HashMap::new();

    for (from_id, list) in &out_edge_order {
        for (to_id, out_idx, out_cnt) in list {
            let key = format!("{}__{}", from_id, to_id);
            let info = edge_port_info.entry(key).or_default();
            info.out_index = *out_idx;
            info.out_count = *out_cnt;
        }
    }
    for (to_id, list) in &in_edge_order {
        for (from_id, in_idx, in_cnt) in list {
            let key = format!("{}__{}", from_id, to_id);
            let info = edge_port_info.entry(key).or_default();
            info.in_index = *in_idx;
            info.in_count = *in_cnt;
        }
    }

    // Build edges
    let mut layout_edges: Vec<LayoutEdge> = Vec::new();
    let mut left_skip_index = 0;
    let mut right_skip_index = 0;

    for e in edges {
        let from_node = match node_map.get(&e.from) {
            Some(n) => n.clone(),
            None => continue,
        };
        let to_node = match node_map.get(&e.to) {
            Some(n) => n.clone(),
            None => continue,
        };

        let from_layer = layer_map.get(&e.from).copied().unwrap_or(0);
        let to_layer = layer_map.get(&e.to).copied().unwrap_or(0);
        let span = to_layer.saturating_sub(from_layer);
        let is_long_range = span > 2;

        if is_long_range {
            let all_nodes: Vec<LayoutNode> = node_map.values().cloned().collect();
            let route_side = choose_side(
                &from_node,
                &to_node,
                &all_nodes,
                from_layer,
                to_layer,
                &layer_map,
                total_width,
            );
            let exit_port = compute_ports(&from_node, true, Some(route_side), false, 0, 1);
            let entry_port = compute_ports(&to_node, true, Some(route_side), true, 0, 1);
            let label = match &e.tensor_name {
                Some(name) => format!("→ {}", name),
                None => "→ skip".to_string(),
            };
            let base_offset = total_width * 0.2 + 60.0;
            let side_index = if route_side == "left" {
                let idx = left_skip_index;
                left_skip_index += 1;
                idx
            } else {
                let idx = right_skip_index;
                right_skip_index += 1;
                idx
            };
            let curve_offset = (base_offset + side_index as f64 * 40.0)
                * if route_side == "left" { -1.0 } else { 1.0 };

            layout_edges.push(LayoutEdge {
                from: from_node,
                to: to_node,
                points: Vec::new(),
                is_long_range: true,
                label: Some(label),
                curve_offset: Some(curve_offset),
                route_side: Some(route_side.to_string()),
                exit_port: Some(exit_port),
                entry_port: Some(entry_port),
                label_position: None,
            });
        } else {
            let port_key = format!("{}__{}", e.from, e.to);
            let port_info = edge_port_info.get(&port_key).copied().unwrap_or_default();
            let out_index = port_info.out_index;
            let out_count = port_info.out_count.max(1);
            let in_index = port_info.in_index;
            let in_count = port_info.in_count.max(1);

            let exit_port = compute_ports(&from_node, false, None, false, out_index, out_count);
            let entry_port = compute_ports(&to_node, false, None, true, in_index, in_count);

            let fx = exit_port.x;
            let fy = exit_port.y;
            let tx = entry_port.x;
            let ty = entry_port.y;

            let points = if (fx - tx).abs() < 1.0 {
                vec![Point { x: fx, y: fy }, Point { x: tx, y: ty }]
            } else {
                let midpoint_stagger = 10.0;
                let base_mid = (fy + ty) / 2.0;
                let mid = if in_count > 1 {
                    base_mid + (in_index as f64 - (in_count as f64 - 1.0) / 2.0) * midpoint_stagger
                } else {
                    base_mid
                };
                vec![
                    Point { x: fx, y: fy },
                    Point { x: fx, y: mid },
                    Point { x: tx, y: mid },
                    Point { x: tx, y: ty },
                ]
            };

            layout_edges.push(LayoutEdge {
                from: from_node,
                to: to_node,
                points,
                is_long_range: false,
                label: e.tensor_name.clone(),
                curve_offset: None,
                route_side: None,
                exit_port: Some(exit_port),
                entry_port: Some(entry_port),
                label_position: None,
            });
        }
    }

    // Compute initial label positions, resolve overlaps, store on edges
    let mut label_rects: Vec<LabelRect> = Vec::with_capacity(layout_edges.len());
    for (i, edge) in layout_edges.iter().enumerate() {
        let (lx, ly) = if edge.is_long_range {
            let p0x = edge
                .exit_port
                .map(|p| p.x)
                .unwrap_or(edge.from.x + edge.from.width);
            let lx = match edge.route_side.as_deref() {
                Some("left") => p0x - 60.0,
                _ => p0x + 8.0,
            };
            let p0y = edge
                .exit_port
                .map(|p| p.y)
                .unwrap_or(edge.from.y + edge.from.height / 2.0);
            (lx, p0y - 12.0)
        } else {
            let pts = &edge.points;
            if pts.len() >= 2 {
                let mid = pts.len() / 2;
                let p1 = if mid > 0 { &pts[mid - 1] } else { &pts[0] };
                let p2 = if mid < pts.len() {
                    &pts[mid]
                } else {
                    &pts[pts.len() - 1]
                };
                ((p1.x + p2.x) / 2.0 + 4.0, (p1.y + p2.y) / 2.0 + 4.0)
            } else {
                (edge.from.x, edge.from.y)
            }
        };
        let label_text = match &edge.label {
            Some(l) => l.clone(),
            None => String::new(),
        };
        let label_width = label_text.len() as f64 * LABEL_CHAR_W + 20.0;
        label_rects.push(LabelRect {
            edge_index: i,
            x: lx,
            y: ly,
            width: label_width,
            height: LABEL_H,
        });
    }

    resolve_overlaps(&mut label_rects, 3);

    for rect in &label_rects {
        if let Some(edge) = layout_edges.get_mut(rect.edge_index) {
            edge.label_position = Some(Point {
                x: rect.x,
                y: rect.y,
            });
        }
    }

    // Adjust height to include group vertical space
    let mut adjusted_height = layout_result_height;
    if !group_layouts.is_empty() {
        let max_group_bottom = group_layouts
            .iter()
            .map(|g| g.y + g.height)
            .fold(0.0_f64, f64::max);
        adjusted_height = adjusted_height.max(max_group_bottom);
    }

    LayoutResult {
        nodes: node_map.into_values().collect(),
        edges: layout_edges,
        groups: group_layouts,
        width: total_width,
        height: adjusted_height,
    }
}

/// Assign each block to a layer (longest path from sources).
fn assign_layers(block_ids: &[String], edges: &[Edge]) -> HashMap<String, usize> {
    let mut layer: HashMap<String, usize> = HashMap::new();
    for id in block_ids {
        layer.insert(id.clone(), 0);
    }
    for id in block_ids {
        let cur = *layer.get(id).unwrap_or(&0);
        for e in edges {
            if e.from == *id {
                let dst = &e.to;
                let dst_val = *layer.get(dst).unwrap_or(&0);
                if dst_val < cur + 1 {
                    layer.insert(dst.clone(), cur + 1);
                }
            }
        }
    }
    layer
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::graph::*;
    use crate::ast::nodes::SourceLoc;
    use std::collections::HashMap;

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
            block_type: "Linear".to_string(),
            params: HashMap::new(),
            input_shapes: vec![],
            output_shapes: vec![],
            param_count: None,
            show_depth: None,
            loc: dummy_loc(),
        }
    }

    fn make_graph(ids: &[&str], edge_pairs: &[(&str, &str)]) -> Graph {
        let blocks: Vec<Block> = ids.iter().map(|id| make_block(id)).collect();
        let edges: Vec<Edge> = edge_pairs
            .iter()
            .map(|(from, to)| Edge {
                from: from.to_string(),
                to: to.to_string(),
                tensor_name: None,
                shape: None,
            })
            .collect();
        Graph {
            blocks,
            edges,
            groups: vec![],
        }
    }

    fn make_graph_with_tensors(
        ids: &[&str],
        edge_pairs: &[(&str, &str)],
        tensor_names: &[(&str, &str)],
    ) -> Graph {
        let blocks: Vec<Block> = ids.iter().map(|id| make_block(id)).collect();
        let tensor_map: std::collections::HashMap<String, String> = tensor_names
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let edges: Vec<Edge> = edge_pairs
            .iter()
            .map(|(from, to)| {
                let key = format!("{}->{}", from, to);
                Edge {
                    from: from.to_string(),
                    to: to.to_string(),
                    tensor_name: tensor_map.get(&key).cloned(),
                    shape: None,
                }
            })
            .collect();
        Graph {
            blocks,
            edges,
            groups: vec![],
        }
    }

    #[test]
    fn test_lenet_linear_chain() {
        let g = make_graph(&["a", "b", "c", "d"], &[("a", "b"), ("b", "c"), ("c", "d")]);
        let result = layout(&g);

        assert_eq!(result.nodes.len(), 4);

        // All nodes in a single column (same x)
        let xs: std::collections::HashSet<u64> = result.nodes.iter().map(|n| n.x as u64).collect();
        assert_eq!(xs.len(), 1);

        // Nodes are vertically ordered
        let by_id: HashMap<&str, &LayoutNode> = result
            .nodes
            .iter()
            .map(|n| (n.block.id.as_str(), n))
            .collect();
        assert!(by_id["a"].y < by_id["b"].y);
        assert!(by_id["b"].y < by_id["c"].y);
        assert!(by_id["c"].y < by_id["d"].y);

        // Edges are not long-range
        assert!(result.edges.iter().all(|e| !e.is_long_range));

        // Straight edges have 2 points (or 4 for orthogonal routing)
        // (they may have 4 points due to orthogonal polyline routing)
    }

    #[test]
    fn test_resnet_fork() {
        // a -> b, a -> c, b -> d, c -> d
        let g = make_graph(
            &["a", "b", "c", "d"],
            &[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        );
        let result = layout(&g);

        let by_id: HashMap<&str, &LayoutNode> = result
            .nodes
            .iter()
            .map(|n| (n.block.id.as_str(), n))
            .collect();

        // b and c are at the same layer (same y)
        assert!((by_id["b"].y - by_id["c"].y).abs() < 0.001);

        // b and c are side-by-side (different x)
        assert!((by_id["b"].x - by_id["c"].x).abs() > 0.001);

        // a and d are in a different layer from b/c
        assert!(by_id["a"].y < by_id["b"].y);
        assert!(by_id["d"].y > by_id["b"].y);
    }

    #[test]
    fn test_unet_skip_long_range() {
        // a -> b -> c -> d -> e, plus skip a -> d
        let g = make_graph_with_tensors(
            &["a", "b", "c", "d", "e"],
            &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "d")],
            &[("a->d", "skip1")],
        );
        let result = layout(&g);

        let skip = result
            .edges
            .iter()
            .find(|e| e.from.block.id == "a" && e.to.block.id == "d");
        assert!(skip.is_some());
        let skip = skip.unwrap();
        assert!(skip.is_long_range);
        assert!(skip.label.is_some());
        assert!(skip.label.as_deref().unwrap().contains("skip1"));
        assert!(skip.points.is_empty());

        assert!(result
            .edges
            .iter()
            .filter(|e| !(e.from.block.id == "a" && e.to.block.id == "d"))
            .all(|e| !e.is_long_range));
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph {
            blocks: vec![],
            edges: vec![],
            groups: vec![],
        };
        let result = layout(&g);
        assert_eq!(result.nodes.len(), 0);
        assert_eq!(result.edges.len(), 0);
        assert_eq!(result.width, 0.0);
        assert_eq!(result.height, 0.0);
    }

    #[test]
    fn test_single_node_dimensions() {
        let g = make_graph(&["a"], &[]);
        let result = layout(&g);
        // Empty outputShapes -> min width (60), min height (28), result height = 28 + V_GAP = 52
        assert!((result.nodes[0].width - 60.0).abs() < 0.001);
        assert!((result.nodes[0].height - 28.0).abs() < 0.001);
        assert!((result.width - 60.0).abs() < 0.001);
        assert!((result.height - 52.0).abs() < 0.001);
    }
}
