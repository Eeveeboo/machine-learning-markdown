use std::collections::HashMap;

// ---------------------------------------------------------------------------
// XML helpers
// ---------------------------------------------------------------------------

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn attrs(opts: Option<&HashMap<String, String>>) -> String {
    let map = match opts {
        Some(m) => m,
        None => return String::new(),
    };
    let mut pairs: Vec<String> = Vec::with_capacity(map.len());
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    for k in keys {
        if let Some(v) = map.get(k) {
            pairs.push(format!(" {}=\"{}\"", k, escape_xml(v)));
        }
    }
    pairs.join("")
}

const ARROWHEAD_MARKER: &str = r##"<defs>
  <marker id="arrowhead" markerWidth="7" markerHeight="5" refX="7" refY="2.5" markerUnits="userSpaceOnUse" orient="auto">
    <polygon points="0 0, 7 2.5, 0 5" fill="#94a3b8" />
  </marker>
</defs>"##;

// ---------------------------------------------------------------------------
// SvgBuilder
// ---------------------------------------------------------------------------

/// A string-based SVG builder (no DOM dependency).
/// Uses the builder pattern: most methods return `&mut Self` for chaining.
/// Groups are created via a closure to keep borrows well-scoped.
pub struct SvgBuilder {
    elements: Vec<String>,
    children: Vec<SvgBuilder>,
    group_id: Option<String>,
    transform: Option<String>,
    width: u32,
    height: u32,
}

impl SvgBuilder {
    /// Create a new root SVG builder with the given canvas dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            elements: Vec::new(),
            children: Vec::new(),
            group_id: None,
            transform: None,
            width,
            height,
        }
    }

    /// Create a non-root builder (used internally by `group`).
    fn new_child(width: u32, height: u32, group_id: String, transform: Option<String>) -> Self {
        Self {
            elements: Vec::new(),
            children: Vec::new(),
            group_id: Some(group_id),
            transform,
            width,
            height,
        }
    }

    // -- Primitive drawing methods --

    /// Add a `<rect>` element.
    pub fn rect(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        opts: Option<HashMap<String, String>>,
    ) -> &mut Self {
        self.elements.push(format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}"{} />"#,
            x,
            y,
            w,
            h,
            attrs(opts.as_ref())
        ));
        self
    }

    /// Add a `<rect>` element with rounded corners (`rx` / `ry`).
    pub fn rounded_rect(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        r: f64,
        opts: Option<HashMap<String, String>>,
    ) -> &mut Self {
        self.elements.push(format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" rx="{}" ry="{}"{} />"#,
            x,
            y,
            w,
            h,
            r,
            r,
            attrs(opts.as_ref())
        ));
        self
    }

    /// Add a `<text>` element.
    pub fn text(
        &mut self,
        x: f64,
        y: f64,
        content: &str,
        opts: Option<HashMap<String, String>>,
    ) -> &mut Self {
        self.elements.push(format!(
            r#"<text x="{}" y="{}"{}>{}</text>"#,
            x,
            y,
            attrs(opts.as_ref()),
            escape_xml(content)
        ));
        self
    }

    /// Add a `<path>` element.
    pub fn path(
        &mut self,
        d: &str,
        opts: Option<HashMap<String, String>>,
    ) -> &mut Self {
        self.elements.push(format!(
            r#"<path d="{}"{} />"#,
            d,
            attrs(opts.as_ref())
        ));
        self
    }

    /// Add a `<line>` element styled as an arrow (with `marker-end`).
    pub fn arrow(
        &mut self,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        opts: Option<HashMap<String, String>>,
    ) -> &mut Self {
        let mut merged = match opts {
            Some(m) => m,
            None => HashMap::new(),
        };
        merged.entry("stroke".to_string()).or_insert_with(|| "#333".to_string());
        merged.entry("stroke-width".to_string()).or_insert_with(|| "2".to_string());
        merged.entry("fill".to_string()).or_insert_with(|| "none".to_string());
        merged
            .entry("marker-end".to_string())
            .or_insert_with(|| "url(#arrowhead)".to_string());

        self.elements.push(format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}"{} />"#,
            x1,
            y1,
            x2,
            y2,
            attrs(Some(&merged))
        ));
        self
    }

    /// Add a `<path>` element shaped as a diamond.
    pub fn diamond(
        &mut self,
        cx: f64,
        cy: f64,
        w: f64,
        h: f64,
        opts: Option<HashMap<String, String>>,
    ) -> &mut Self {
        let hw = w / 2.0;
        let hh = h / 2.0;
        let d = format!(
            "M {} {} L {} {} L {} {} L {} {} Z",
            cx,
            cy - hh,
            cx + hw,
            cy,
            cx,
            cy + hh,
            cx - hw,
            cy
        );
        self.elements.push(format!(
            r#"<path d="{}"{} />"#,
            d,
            attrs(opts.as_ref())
        ));
        self
    }

    /// Create a child `<g>` element with the given `id` and optional `transform`.
    ///
    /// The closure `f` receives a mutable reference to the child builder so that
    /// elements and nested groups can be added inside the group scope.
    pub fn group<F>(&mut self, id: &str, transform: Option<&str>, f: F) -> &mut Self
    where
        F: FnOnce(&mut SvgBuilder),
    {
        let mut child = SvgBuilder::new_child(
            self.width,
            self.height,
            id.to_string(),
            transform.map(|s| s.to_string()),
        );
        f(&mut child);
        self.children.push(child);
        // Push a placeholder that will be replaced during rendering
        self.elements
            .push(format!("__child_{}__", self.children.len() - 1));
        self
    }

    // -- Rendering --

    /// Render all elements, replacing child placeholders with rendered group markup.
    fn render(&self) -> String {
        let mut child_idx = 0;
        let mut lines: Vec<String> = Vec::with_capacity(self.elements.len());
        for el in &self.elements {
            if el.starts_with("__child_") && el.ends_with("__") {
                let idx = child_idx;
                child_idx += 1;
                if idx < self.children.len() {
                    lines.push(self.children[idx].render_group());
                }
            } else {
                lines.push(el.clone());
            }
        }
        lines.join("\n")
    }

    /// Render this builder as an SVG `<g>` element (with id and optional transform).
    fn render_group(&self) -> String {
        let transform_attr = match &self.transform {
            Some(t) => format!(" transform=\"{}\"", escape_xml(t)),
            None => String::new(),
        };
        let inner = self.render();
        format!(
            "<g id=\"{}\"{}>\n{}\n</g>",
            escape_xml(self.group_id.as_deref().unwrap_or("")),
            transform_attr,
            inner
        )
    }

    /// Render the full SVG document string, including the namespace, viewBox,
    /// arrowhead marker definitions, and all elements.
    pub fn to_string(&self) -> String {
        let inner = self.render();
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
{}
{}
</svg>"#,
            self.width,
            self.height,
            self.width,
            self.height,
            ARROWHEAD_MARKER,
            inner
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svg_document_tags() {
        let svg = SvgBuilder::new(800, 600);
        let out = svg.to_string();
        assert!(out.contains("<svg"));
        assert!(out.contains("</svg>"));
        assert!(out.contains("xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(out.contains("viewBox"));
    }

    #[test]
    fn test_rect() {
        let mut svg = SvgBuilder::new(800, 600);
        svg.rect(10.0, 20.0, 100.0, 50.0, None);
        let out = svg.to_string();
        assert!(out.contains("<rect"));
        assert!(out.contains("x=\"10\""));
        assert!(out.contains("y=\"20\""));
        assert!(out.contains("width=\"100\""));
        assert!(out.contains("height=\"50\""));
    }

    #[test]
    fn test_rounded_rect() {
        let mut svg = SvgBuilder::new(800, 600);
        svg.rounded_rect(0.0, 0.0, 80.0, 40.0, 8.0, None);
        let out = svg.to_string();
        assert!(out.contains("rx=\"8\""));
        assert!(out.contains("ry=\"8\""));
    }

    #[test]
    fn test_diamond() {
        let mut svg = SvgBuilder::new(800, 600);
        svg.diamond(50.0, 50.0, 60.0, 40.0, None);
        let out = svg.to_string();
        assert!(out.contains("<path"));
    }

    #[test]
    fn test_text() {
        let mut svg = SvgBuilder::new(800, 600);
        svg.text(5.0, 15.0, "hello", None);
        let out = svg.to_string();
        assert!(out.contains("<text"));
        assert!(out.contains("hello"));
    }

    #[test]
    fn test_path() {
        let mut svg = SvgBuilder::new(800, 600);
        svg.path("M 0 0 L 100 100", None);
        let out = svg.to_string();
        assert!(out.contains("<path"));
        assert!(out.contains("M 0 0 L 100 100"));
    }

    #[test]
    fn test_arrow() {
        let mut svg = SvgBuilder::new(800, 600);
        svg.arrow(0.0, 0.0, 100.0, 100.0, None);
        let out = svg.to_string();
        assert!(out.contains("<line"));
        assert!(out.contains("marker-end"));
        assert!(out.contains("arrowhead"));
        assert!(out.contains("<marker"));
    }

    #[test]
    fn test_group_nesting() {
        let mut svg = SvgBuilder::new(800, 600);
        svg.group("layer1", None, |g| {
            g.rect(0.0, 0.0, 50.0, 50.0, None);
            g.group("layer2", Some("translate(10,10)"), |g2| {
                g2.text(0.0, 0.0, "nested", None);
            });
        });
        let out = svg.to_string();
        assert!(out.contains("<g id=\"layer1\""));
        assert!(out.contains("<g id=\"layer2\""));
        assert!(out.contains("translate(10,10)"));
        assert!(out.contains("nested"));
    }

    #[test]
    fn test_opts_rendered_as_attributes() {
        let mut svg = SvgBuilder::new(800, 600);
        let mut opts = HashMap::new();
        opts.insert("fill".to_string(), "red".to_string());
        opts.insert("stroke".to_string(), "blue".to_string());
        svg.rect(0.0, 0.0, 10.0, 10.0, Some(opts));
        let out = svg.to_string();
        assert!(out.contains("fill=\"red\""));
        assert!(out.contains("stroke=\"blue\""));
    }

    #[test]
    fn test_method_chaining() {
        let mut svg = SvgBuilder::new(800, 600);
        svg.rect(0.0, 0.0, 10.0, 10.0, None)
            .text(5.0, 5.0, "x", None)
            .path("M0 0", None);
        // No panic means success
    }

    #[test]
    fn test_escape_xml_in_content() {
        let mut svg = SvgBuilder::new(800, 600);
        svg.text(0.0, 0.0, "a < b & c > d", None);
        let out = svg.to_string();
        assert!(out.contains("a &lt; b &amp; c &gt; d"));
    }

    #[test]
    fn test_escape_xml_in_attr() {
        let mut opts = HashMap::new();
        opts.insert("data-label".to_string(), "a < b".to_string());
        let mut svg = SvgBuilder::new(800, 600);
        svg.rect(0.0, 0.0, 10.0, 10.0, Some(opts));
        let out = svg.to_string();
        assert!(out.contains("data-label=\"a &lt; b\""));
    }
}
