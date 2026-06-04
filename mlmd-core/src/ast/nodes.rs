use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Source location
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceLoc {
    pub line: usize,
    pub col: usize,
    pub offset: usize,
}

// ---------------------------------------------------------------------------
// Value types used as parameter values in the AST
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberVal {
    pub kind: String,
    pub value: f64,
    pub loc: SourceLoc,
}

impl NumberVal {
    pub fn new(value: f64, loc: SourceLoc) -> Self {
        Self {
            kind: "number".to_string(),
            value,
            loc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringVal {
    pub kind: String,
    pub value: String,
    pub loc: SourceLoc,
}

impl StringVal {
    pub fn new(value: String, loc: SourceLoc) -> Self {
        Self {
            kind: "string".to_string(),
            value,
            loc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoolVal {
    pub kind: String,
    pub value: bool,
    pub loc: SourceLoc,
}

impl BoolVal {
    pub fn new(value: bool, loc: SourceLoc) -> Self {
        Self {
            kind: "bool".to_string(),
            value,
            loc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarewordVal {
    pub kind: String,
    pub value: String,
    pub loc: SourceLoc,
}

impl BarewordVal {
    pub fn new(value: String, loc: SourceLoc) -> Self {
        Self {
            kind: "bareword".to_string(),
            value,
            loc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeVal {
    pub kind: String,
    pub dims: Vec<usize>,
    pub loc: SourceLoc,
}

impl ShapeVal {
    pub fn new(dims: Vec<usize>, loc: SourceLoc) -> Self {
        Self {
            kind: "shape".to_string(),
            dims,
            loc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListVal {
    pub kind: String,
    pub items: Vec<ParamValue>,
    pub loc: SourceLoc,
}

impl ListVal {
    pub fn new(items: Vec<ParamValue>, loc: SourceLoc) -> Self {
        Self {
            kind: "list".to_string(),
            items,
            loc,
        }
    }
}

// ---------------------------------------------------------------------------
// ParamValue enum — discriminated union matching the TS union type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamValue {
    Number(Box<NumberVal>),
    String(Box<StringVal>),
    Bool(Box<BoolVal>),
    Bareword(Box<BarewordVal>),
    Shape(Box<ShapeVal>),
    List(Box<ListVal>),
}

impl ParamValue {
    /// Return a human-readable type name for this value.
    pub fn type_name(&self) -> &str {
        match self {
            ParamValue::Number(_) => "number",
            ParamValue::String(_) => "string",
            ParamValue::Bool(_) => "bool",
            ParamValue::Bareword(_) => "bareword",
            ParamValue::Shape(_) => "shape",
            ParamValue::List(_) => "list",
        }
    }

    /// Return a reference to the source location.
    pub fn loc(&self) -> &SourceLoc {
        match self {
            ParamValue::Number(v) => &v.loc,
            ParamValue::String(v) => &v.loc,
            ParamValue::Bool(v) => &v.loc,
            ParamValue::Bareword(v) => &v.loc,
            ParamValue::Shape(v) => &v.loc,
            ParamValue::List(v) => &v.loc,
        }
    }
}

impl fmt::Display for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamValue::Number(v) => write!(f, "{}", v.value),
            ParamValue::String(v) => write!(f, "\"{}\"", v.value),
            ParamValue::Bool(v) => write!(f, "{}", v.value),
            ParamValue::Bareword(v) => write!(f, "{}", v.value),
            ParamValue::Shape(v) => write!(f, "{:?}", v.dims),
            ParamValue::List(v) => {
                write!(f, "[")?;
                for (i, item) in v.items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Param (name + value + loc)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub value: ParamValue,
    pub loc: SourceLoc,
}

// ---------------------------------------------------------------------------
// AST node types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockDecl {
    pub kind: String,
    pub block_type: String,
    pub params: Vec<Param>,
    pub loc: SourceLoc,
}

impl BlockDecl {
    pub fn new(block_type: String, params: Vec<Param>, loc: SourceLoc) -> Self {
        Self {
            kind: "block".to_string(),
            block_type,
            params,
            loc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorName {
    pub kind: String,
    pub names: Vec<String>,
    pub loc: SourceLoc,
}

impl TensorName {
    pub fn new(names: Vec<String>, loc: SourceLoc) -> Self {
        Self {
            kind: "tensor_name".to_string(),
            names,
            loc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorJoin {
    pub kind: String,
    pub sources: Vec<String>,
    pub target: BlockDecl,
    pub loc: SourceLoc,
}

impl TensorJoin {
    pub fn new(sources: Vec<String>, target: BlockDecl, loc: SourceLoc) -> Self {
        Self {
            kind: "tensor_join".to_string(),
            sources,
            target,
            loc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupDecl {
    pub kind: String,
    pub path: Vec<String>,
    pub body: Vec<ASTNode>,
    pub loc: SourceLoc,
}

impl GroupDecl {
    pub fn new(path: Vec<String>, body: Vec<ASTNode>, loc: SourceLoc) -> Self {
        Self {
            kind: "group".to_string(),
            path,
            body,
            loc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub kind: String,
    pub text: String,
    pub loc: SourceLoc,
}

impl Comment {
    pub fn new(text: String, loc: SourceLoc) -> Self {
        Self {
            kind: "comment".to_string(),
            text,
            loc,
        }
    }
}

// ---------------------------------------------------------------------------
// ASTNode — top-level discriminated union
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ASTNode {
    Block(Box<BlockDecl>),
    TensorName(Box<TensorName>),
    TensorJoin(Box<TensorJoin>),
    Group(Box<GroupDecl>),
    Comment(Box<Comment>),
}

impl ASTNode {
    /// Return the source location for this node.
    pub fn loc(&self) -> &SourceLoc {
        match self {
            ASTNode::Block(n) => &n.loc,
            ASTNode::TensorName(n) => &n.loc,
            ASTNode::TensorJoin(n) => &n.loc,
            ASTNode::Group(n) => &n.loc,
            ASTNode::Comment(n) => &n.loc,
        }
    }
}
