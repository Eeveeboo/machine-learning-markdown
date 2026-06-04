use std::collections::HashMap;

use crate::ast::graph::Graph;
use crate::ast::nodes::ParamValue;

use super::layout::{LayoutResult, Point};
use super::svg_builder::SvgBuilder;

// ---------------------------------------------------------------------------
// Block category detection
// ---------------------------------------------------------------------------

const MERGE_TYPES: &[&str] = &["Add", "Mul", "Sub", "Div", "Concat", "MatMul"];
const ACTIVATION_TYPES: &[&str] = &[
    "ReLU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "ELU",
    "Swish",
    "Mish",
    "Softmax",
    "LogSoftmax",
];

fn block_category(block_type: &str) -> &'static str {
    if MERGE_TYPES.contains(&block_type) {
        "merge"
    } else if ACTIVATION_TYPES.contains(&block_type) {
        "activation"
    } else {
        "default"
    }
}

fn block_colors(block_type: &str) -> (String, String) {
    match block_category(block_type) {
        "merge" => ("#ede9fe".to_string(), "#7c3aed".to_string()),
        "activation" => ("#dcfce7".to_string(), "#16a34a".to_string()),
        _ => ("#dbeafe".to_string(), "#2563eb".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Param formatting
// ---------------------------------------------------------------------------

/// Format a ParamValue as a display string.
fn format_param_value(v: &ParamValue) -> String {
    match v {
        ParamValue::Number(n) => format!("{}", n.value),
        ParamValue::String(s) => format!("\"{}\"", s.value),
        ParamValue::Bool(b) => format!("{}", b.value),
        ParamValue::Bareword(bw) => bw.value.clone(),
        ParamValue::Shape(s) => format!(
            "({})",
            s.dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ),
        ParamValue::List(l) => {
            let items: Vec<String> = l.items.iter().map(format_param_value).collect();
            format!("[{}]", items.join(","))
        }
    }
}

/// Build a short param summary like "k=5 f=6".
fn param_summary(params: &HashMap<String, ParamValue>) -> String {
    let short: HashMap<&str, &str> = [
        ("filters", "f"),
        ("kernel", "k"),
        ("kernel_size", "k"),
        ("out_features", "out"),
        ("in_features", "in"),
        ("stride", "s"),
        ("padding", "p"),
        ("groups", "g"),
        ("num_heads", "h"),
        ("dropout", "drop"),
    ]
    .iter()
    .cloned()
    .collect();

    let parts: Vec<String> = params
        .iter()
        .map(|(k, v)| {
            let key = short.get(k.as_str()).copied().unwrap_or(k.as_str());
            format!("{}={}", key, format_param_value(v))
        })
        .collect();
    parts.join(" ")
}

fn shape_label(shape: &[usize]) -> String {
    format!(
        "({})",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Format number with K/M suffix.
fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

const PADDING: f64 = 40.0;

// ---------------------------------------------------------------------------
// Main render function
// ---------------------------------------------------------------------------

/// Generate an SVG string from a graph and its layout.
pub fn render(graph: &Graph, layout_result: &LayoutResult) -> String {
    let nodes = &layout_result.nodes;
    let edges = &layout_result.edges;
    let groups = &layout_result.groups;
    let width = layout_result.width;
    let height = layout_result.height;

    if nodes.is_empty() {
        let mut svg = SvgBuilder::new(200, 80);
        let mut opts = HashMap::new();
        opts.insert("font-size".to_string(), "14".to_string());
        opts.insert("fill".to_string(), "#666".to_string());
        svg.text(20.0, 40.0, "Empty graph", Some(opts));
        return svg.to_string();
    }

    let canvas_w = width + PADDING * 2.0;
    let canvas_h = height + PADDING * 2.0;

    // Account for long-range bezier curves extending past the right or left edge
    let max_curve_offset = edges
        .iter()
        .filter(|e| e.is_long_range)
        .map(|e| e.curve_offset.unwrap_or(0.0))
        .fold(0.0_f64, f64::max);
    let min_curve_offset = edges
        .iter()
        .filter(|e| e.is_long_range)
        .map(|e| e.curve_offset.unwrap_or(0.0))
        .fold(0.0_f64, f64::min);
    let left_extension = min_curve_offset.abs().min(0.0_f64).abs();
    let adjusted_w = canvas_w + max_curve_offset + left_extension;
    let x_offset = PADDING + left_extension;

    let mut svg = SvgBuilder::new(adjusted_w as u32, canvas_h as u32);

    // --- Groups (dashed bounding boxes) ---
    for group in groups {
        let gx = group.x + x_offset;
        let gy = group.y + PADDING;

        let mut rect_opts = HashMap::new();
        rect_opts.insert("fill".to_string(), "none".to_string());
        rect_opts.insert("stroke".to_string(), "#94a3b8".to_string());
        rect_opts.insert("stroke-width".to_string(), "1.5".to_string());
        rect_opts.insert("stroke-dasharray".to_string(), "6,4".to_string());
        rect_opts.insert("rx".to_string(), "8".to_string());
        svg.rect(gx, gy, group.width, group.height, Some(rect_opts));

        let mut text_opts = HashMap::new();
        text_opts.insert("font-size".to_string(), "11".to_string());
        text_opts.insert("fill".to_string(), "#64748b".to_string());
        text_opts.insert("font-family".to_string(), "sans-serif".to_string());
        text_opts.insert("font-weight".to_string(), "500".to_string());
        svg.text(
            gx + 8.0,
            gy + 14.0,
            &group.path.join(" / "),
            Some(text_opts),
        );
    }

    // --- Nodes ---
    for node in nodes {
        let px = node.x + x_offset;
        let py = node.y + PADDING;
        let cx = px + node.width / 2.0;
        let (fill, stroke) = block_colors(&node.block.block_type);

        // All blocks use rounded rects
        let mut rect_opts = HashMap::new();
        rect_opts.insert("fill".to_string(), fill);
        rect_opts.insert("stroke".to_string(), stroke);
        rect_opts.insert("stroke-width".to_string(), "2".to_string());
        svg.rounded_rect(px, py, node.width, node.height, 8.0, Some(rect_opts));

        // Block type label
        let summary = param_summary(&node.block.params);
        let label = if summary.is_empty() {
            node.block.block_type.clone()
        } else {
            format!("{} {}", node.block.block_type, summary)
        };

        let mut type_opts = HashMap::new();
        type_opts.insert("text-anchor".to_string(), "middle".to_string());
        type_opts.insert("font-size".to_string(), "11".to_string());
        type_opts.insert("fill".to_string(), "#1e293b".to_string());
        type_opts.insert("font-family".to_string(), "sans-serif".to_string());
        type_opts.insert("font-weight".to_string(), "500".to_string());
        svg.text(cx, py + 14.0, &label, Some(type_opts));

        // Output shape label(s) — all outputs for multi-output blocks
        let shape_labels: Vec<String> = node
            .block
            .output_shapes
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| shape_label(s))
            .collect();
        if !shape_labels.is_empty() {
            let mut shape_opts = HashMap::new();
            shape_opts.insert("text-anchor".to_string(), "middle".to_string());
            shape_opts.insert("font-size".to_string(), "8".to_string());
            shape_opts.insert("fill".to_string(), "#64748b".to_string());
            shape_opts.insert("font-family".to_string(), "monospace".to_string());
            svg.text(
                cx,
                py + node.height - 4.0,
                &shape_labels.join(" | "),
                Some(shape_opts),
            );
        }
    }

    // --- Edges ---
    // Find edges with shape info from graph.edges
    let mut graph_edge_map: HashMap<String, (Option<Vec<usize>>, Option<String>)> = HashMap::new();
    for e in &graph.edges {
        let key = format!("{}__{}", e.from, e.to);
        graph_edge_map.insert(key, (e.shape.clone(), e.tensor_name.clone()));
    }

    for edge in edges {
        let key = format!("{}__{}", edge.from.block.id, edge.to.block.id);
        let graph_edge = graph_edge_map.get(&key);
        let shape_str = graph_edge
            .and_then(|(shape, _)| shape.as_ref())
            .filter(|s| !s.is_empty())
            .map(|s| shape_label(s));

        // Build the label parts for this edge
        let mut edge_label_parts: Vec<String> = Vec::new();
        if edge.is_long_range {
            if let Some(ref label) = edge.label {
                edge_label_parts.push(label.clone());
            }
        } else if let Some((_, Some(tensor_name))) = graph_edge {
            edge_label_parts.push(tensor_name.clone());
        }
        if let Some(ref s) = shape_str {
            edge_label_parts.push(s.clone());
        }
        let edge_label = edge_label_parts.join(" ");

        if edge.is_long_range {
            // Bezier curve using exit/entry ports
            let p0 = edge.exit_port.unwrap_or(Point {
                x: edge.from.x + edge.from.width,
                y: edge.from.y + edge.from.height / 2.0,
            });
            let p3 = edge.entry_port.unwrap_or(Point {
                x: edge.to.x + edge.to.width,
                y: edge.to.y + edge.to.height / 2.0,
            });
            let off = edge.curve_offset.unwrap_or(80.0);
            let p1 = Point {
                x: p0.x + off,
                y: p0.y,
            };
            let p2 = Point {
                x: p3.x + off,
                y: p3.y,
            };
            let d = format!(
                "M {} {} C {} {}, {} {}, {} {}",
                p0.x + x_offset,
                p0.y + PADDING,
                p1.x + x_offset,
                p1.y + PADDING,
                p2.x + x_offset,
                p2.y + PADDING,
                p3.x + x_offset,
                p3.y + PADDING
            );

            let mut path_opts = HashMap::new();
            path_opts.insert("stroke".to_string(), "#9333ea".to_string());
            path_opts.insert("stroke-width".to_string(), "1.5".to_string());
            path_opts.insert("fill".to_string(), "none".to_string());
            path_opts.insert("stroke-dasharray".to_string(), "5,3".to_string());
            path_opts.insert("marker-end".to_string(), "url(#arrowhead)".to_string());
            svg.path(&d, Some(path_opts));

            // Label: position near the exit port
            let lx = edge
                .label_position
                .map(|p| p.x + x_offset)
                .unwrap_or(p0.x + 8.0 + x_offset);
            let ly = edge
                .label_position
                .map(|p| p.y + PADDING)
                .unwrap_or(p0.y - 6.0 + PADDING);

            let mut label_opts = HashMap::new();
            label_opts.insert("font-size".to_string(), "9".to_string());
            label_opts.insert("fill".to_string(), "#9333ea".to_string());
            label_opts.insert(
                "text-anchor".to_string(),
                match edge.route_side.as_deref() {
                    Some("left") => "end",
                    _ => "start",
                }
                .to_string(),
            );
            label_opts.insert("font-family".to_string(), "monospace".to_string());
            svg.text(lx, ly, &edge_label, Some(label_opts));
        } else if edge.points.len() >= 2 {
            // Build polyline path
            let pts = &edge.points;
            let mut d = format!("M {} {}", pts[0].x + x_offset, pts[0].y + PADDING);
            for p in &pts[1..] {
                d += &format!(" L {} {}", p.x + x_offset, p.y + PADDING);
            }

            let mut path_opts = HashMap::new();
            path_opts.insert("stroke".to_string(), "#64748b".to_string());
            path_opts.insert("stroke-width".to_string(), "1.5".to_string());
            path_opts.insert("fill".to_string(), "none".to_string());
            path_opts.insert("marker-end".to_string(), "url(#arrowhead)".to_string());
            svg.path(&d, Some(path_opts));

            // Label centered in vertical gap between blocks
            if !edge_label.is_empty() {
                let mid = pts.len() / 2;
                let p1 = if mid > 0 { &pts[mid - 1] } else { &pts[0] };
                let p2 = if mid < pts.len() {
                    &pts[mid]
                } else {
                    &pts[pts.len() - 1]
                };
                let lx = edge
                    .label_position
                    .map(|p| p.x + x_offset)
                    .unwrap_or((p1.x + p2.x) / 2.0 + 4.0 + x_offset);
                let ly = edge
                    .label_position
                    .map(|p| p.y + PADDING)
                    .unwrap_or((p1.y + p2.y) / 2.0 + 4.0 + PADDING);

                let mut label_opts = HashMap::new();
                label_opts.insert("font-size".to_string(), "8".to_string());
                label_opts.insert("fill".to_string(), "#94a3b8".to_string());
                label_opts.insert("font-family".to_string(), "monospace".to_string());
                svg.text(lx, ly, &edge_label, Some(label_opts));
            }
        }
    }

    // --- Total parameter count ---
    let mut total_params: usize = 0;
    for node in nodes {
        if let Some(count) = node.block.param_count {
            total_params += count;
        }
    }
    let mut params_opts = HashMap::new();
    params_opts.insert("text-anchor".to_string(), "end".to_string());
    params_opts.insert("font-size".to_string(), "10".to_string());
    params_opts.insert("fill".to_string(), "#94a3b8".to_string());
    params_opts.insert("font-family".to_string(), "sans-serif".to_string());
    svg.text(
        canvas_w - 10.0,
        canvas_h - 6.0,
        &format!("Total params: {}", format_count(total_params)),
        Some(params_opts),
    );

    svg.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::graph::*;
    use crate::ast::nodes::SourceLoc;
    use crate::visualize::layout::layout;

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
            input_shapes: vec![],
            output_shapes: vec![],
            param_count: None,
            show_depth: None,
            loc: dummy_loc(),
        }
    }

    fn make_block_with_shapes(
        id: &str,
        block_type: &str,
        params: HashMap<String, ParamValue>,
        output_shapes: Vec<Vec<usize>>,
    ) -> Block {
        Block {
            id: id.to_string(),
            block_type: block_type.to_string(),
            params,
            input_shapes: vec![],
            output_shapes,
            param_count: None,
            show_depth: None,
            loc: dummy_loc(),
        }
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph {
            blocks: vec![],
            edges: vec![],
            groups: vec![],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_single_block() {
        let blocks = vec![make_block_with_shapes(
            "a",
            "Conv2d",
            {
                let mut m = HashMap::new();
                m.insert(
                    "filters".to_string(),
                    ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                        6.0,
                        dummy_loc(),
                    ))),
                );
                m.insert(
                    "kernel".to_string(),
                    ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                        5.0,
                        dummy_loc(),
                    ))),
                );
                m
            },
            vec![vec![6, 24, 24]],
        )];
        let g = Graph {
            blocks,
            edges: vec![],
            groups: vec![],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<text"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Conv2d"));
    }

    #[test]
    fn test_merge_block_add() {
        let blocks = vec![make_block("a", "Add")];
        let g = Graph {
            blocks,
            edges: vec![],
            groups: vec![],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("<rect"));
        assert!(svg.contains("#ede9fe"));
        assert!(svg.contains("#7c3aed"));
        assert!(svg.contains("Add"));
    }

    #[test]
    fn test_activation_relu() {
        let blocks = vec![make_block("a", "ReLU")];
        let g = Graph {
            blocks,
            edges: vec![],
            groups: vec![],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("#dcfce7"));
    }

    #[test]
    fn test_edge_with_shape_label() {
        let blocks = vec![
            make_block_with_shapes("a", "Conv2d", HashMap::new(), vec![vec![6, 24, 24]]),
            make_block_with_shapes("b", "ReLU", HashMap::new(), vec![vec![6, 24, 24]]),
        ];
        let g = Graph {
            blocks,
            edges: vec![Edge {
                from: "a".to_string(),
                to: "b".to_string(),
                tensor_name: None,
                shape: Some(vec![6, 24, 24]),
            }],
            groups: vec![],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("6,24,24"));
    }

    #[test]
    fn test_chain_produces_arrows() {
        let blocks = vec![
            make_block_with_shapes(
                "a",
                "Linear",
                {
                    let mut m = HashMap::new();
                    m.insert(
                        "out_features".to_string(),
                        ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                            120.0,
                            dummy_loc(),
                        ))),
                    );
                    m
                },
                vec![vec![120]],
            ),
            make_block("b", "ReLU"),
            make_block_with_shapes(
                "c",
                "Linear",
                {
                    let mut m = HashMap::new();
                    m.insert(
                        "out_features".to_string(),
                        ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                            10.0,
                            dummy_loc(),
                        ))),
                    );
                    m
                },
                vec![vec![10]],
            ),
        ];
        let g = Graph {
            blocks,
            edges: vec![
                Edge {
                    from: "a".to_string(),
                    to: "b".to_string(),
                    tensor_name: None,
                    shape: None,
                },
                Edge {
                    from: "b".to_string(),
                    to: "c".to_string(),
                    tensor_name: None,
                    shape: None,
                },
            ],
            groups: vec![],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("arrowhead"));
        assert!(svg.contains("Linear"));
        assert!(svg.contains("ReLU"));
    }

    #[test]
    fn test_group_bounding_box() {
        let blocks = vec![make_block("a", "Conv2d"), make_block("b", "ReLU")];
        let g = Graph {
            blocks,
            edges: vec![Edge {
                from: "a".to_string(),
                to: "b".to_string(),
                tensor_name: None,
                shape: None,
            }],
            groups: vec![Group {
                path: vec!["MyModel".to_string()],
                block_ids: vec!["a".to_string(), "b".to_string()],
            }],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("stroke-dasharray"));
        assert!(svg.contains("MyModel"));
    }

    #[test]
    fn test_named_edge_label() {
        let block_a = make_block_with_shapes(
            "a",
            "Conv2d",
            {
                let mut m = HashMap::new();
                m.insert(
                    "filters".to_string(),
                    ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                        32.0,
                        dummy_loc(),
                    ))),
                );
                m
            },
            vec![vec![32, 64, 64]],
        );
        let block_b = make_block_with_shapes("b", "ReLU", HashMap::new(), vec![vec![32, 64, 64]]);
        let g = Graph {
            blocks: vec![block_a, block_b],
            edges: vec![Edge {
                from: "a".to_string(),
                to: "b".to_string(),
                tensor_name: Some("enc1".to_string()),
                shape: Some(vec![32, 64, 64]),
            }],
            groups: vec![],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("enc1"));
        assert!(svg.contains("32,64,64"));
    }

    #[test]
    fn test_long_range_edge() {
        // Create a long chain so the edge a->f is long-range
        let blocks = vec![
            make_block("a", "Linear"),
            make_block("b", "Linear"),
            make_block("c", "Linear"),
            make_block("d", "Linear"),
            make_block("e", "Linear"),
            make_block("f", "Linear"),
        ];
        let g = Graph {
            blocks,
            edges: vec![
                Edge {
                    from: "a".to_string(),
                    to: "b".to_string(),
                    tensor_name: None,
                    shape: None,
                },
                Edge {
                    from: "b".to_string(),
                    to: "c".to_string(),
                    tensor_name: None,
                    shape: None,
                },
                Edge {
                    from: "c".to_string(),
                    to: "d".to_string(),
                    tensor_name: None,
                    shape: None,
                },
                Edge {
                    from: "d".to_string(),
                    to: "e".to_string(),
                    tensor_name: None,
                    shape: None,
                },
                Edge {
                    from: "e".to_string(),
                    to: "f".to_string(),
                    tensor_name: None,
                    shape: None,
                },
                Edge {
                    from: "a".to_string(),
                    to: "f".to_string(),
                    tensor_name: Some("skip".to_string()),
                    shape: None,
                },
            ],
            groups: vec![],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("skip"));
        assert!(svg.contains(" C ")); // cubic bezier command
    }

    #[test]
    fn test_lenet_style_chain() {
        let blocks = vec![
            make_block_with_shapes(
                "b0",
                "Input",
                {
                    let mut m = HashMap::new();
                    m.insert(
                        "shape".to_string(),
                        ParamValue::Shape(Box::new(crate::ast::nodes::ShapeVal::new(
                            vec![1, 28, 28],
                            dummy_loc(),
                        ))),
                    );
                    m
                },
                vec![vec![1, 28, 28]],
            ),
            make_block_with_shapes(
                "b1",
                "Conv2d",
                {
                    let mut m = HashMap::new();
                    m.insert(
                        "filters".to_string(),
                        ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                            6.0,
                            dummy_loc(),
                        ))),
                    );
                    m.insert(
                        "kernel".to_string(),
                        ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                            5.0,
                            dummy_loc(),
                        ))),
                    );
                    m
                },
                vec![vec![6, 24, 24]],
            ),
            make_block("b2", "ReLU"),
            make_block_with_shapes(
                "b3",
                "MaxPool",
                {
                    let mut m = HashMap::new();
                    m.insert(
                        "kernel".to_string(),
                        ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                            2.0,
                            dummy_loc(),
                        ))),
                    );
                    m.insert(
                        "stride".to_string(),
                        ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                            2.0,
                            dummy_loc(),
                        ))),
                    );
                    m
                },
                vec![vec![6, 12, 12]],
            ),
            make_block("b4", "Flatten"),
            make_block_with_shapes(
                "b5",
                "Linear",
                {
                    let mut m = HashMap::new();
                    m.insert(
                        "out_features".to_string(),
                        ParamValue::Number(Box::new(crate::ast::nodes::NumberVal::new(
                            120.0,
                            dummy_loc(),
                        ))),
                    );
                    m
                },
                vec![vec![120]],
            ),
        ];
        let g = Graph {
            blocks,
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
                Edge {
                    from: "b2".to_string(),
                    to: "b3".to_string(),
                    tensor_name: None,
                    shape: None,
                },
                Edge {
                    from: "b3".to_string(),
                    to: "b4".to_string(),
                    tensor_name: None,
                    shape: None,
                },
                Edge {
                    from: "b4".to_string(),
                    to: "b5".to_string(),
                    tensor_name: None,
                    shape: None,
                },
            ],
            groups: vec![],
        };
        let result = layout(&g);
        let svg = render(&g, &result);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<text"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Conv2d"));
        assert!(svg.contains("ReLU"));
        assert!(svg.contains("Linear"));
    }
}
