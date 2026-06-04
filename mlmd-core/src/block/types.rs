use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ast::graph::Shape;
use crate::ast::nodes::ParamValue;

// ---------------------------------------------------------------------------
// ParamType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParamType {
    Number,
    String,
    Bool,
    Shape,
    List,
    Bareword,
}

impl ParamType {
    /// Parse a ParamType from a string slice.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "number" => Some(ParamType::Number),
            "string" => Some(ParamType::String),
            "bool" => Some(ParamType::Bool),
            "shape" => Some(ParamType::Shape),
            "list" => Some(ParamType::List),
            "bareword" => Some(ParamType::Bareword),
            _ => None,
        }
    }

    /// Return the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            ParamType::Number => "number",
            ParamType::String => "string",
            ParamType::Bool => "bool",
            ParamType::Shape => "shape",
            ParamType::List => "list",
            ParamType::Bareword => "bareword",
        }
    }
}

// ---------------------------------------------------------------------------
// ParamSpec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSpec {
    pub name: String,
    pub param_type: ParamType,
    pub required: bool,
    pub default: Option<ParamValue>,
}

// ---------------------------------------------------------------------------
// ParamSpecBuilder — ergonomic builder for ParamSpec
// ---------------------------------------------------------------------------

/// Builder for constructing `ParamSpec` values concisely.
///
/// Created via the static methods on `ParamSpec`:
///
/// ```ignore
/// let ps = ParamSpec::number("dropout").default_num(0.5);
/// let pr = ParamSpec::string("activation").required();
/// ```
pub struct ParamSpecBuilder {
    name: String,
    param_type: ParamType,
}

impl ParamSpec {
    /// Create a builder for a number parameter.
    pub fn number(name: impl Into<String>) -> ParamSpecBuilder {
        ParamSpecBuilder {
            name: name.into(),
            param_type: ParamType::Number,
        }
    }

    /// Create a builder for a string parameter.
    pub fn string(name: impl Into<String>) -> ParamSpecBuilder {
        ParamSpecBuilder {
            name: name.into(),
            param_type: ParamType::String,
        }
    }

    /// Create a builder for a boolean parameter.
    pub fn boolean(name: impl Into<String>) -> ParamSpecBuilder {
        ParamSpecBuilder {
            name: name.into(),
            param_type: ParamType::Bool,
        }
    }

    /// Create a builder for a shape parameter.
    pub fn shape(name: impl Into<String>) -> ParamSpecBuilder {
        ParamSpecBuilder {
            name: name.into(),
            param_type: ParamType::Shape,
        }
    }

    /// Create a builder for a bareword parameter.
    pub fn bareword(name: impl Into<String>) -> ParamSpecBuilder {
        ParamSpecBuilder {
            name: name.into(),
            param_type: ParamType::Bareword,
        }
    }

    /// Create a builder for a list parameter.
    pub fn list(name: impl Into<String>) -> ParamSpecBuilder {
        ParamSpecBuilder {
            name: name.into(),
            param_type: ParamType::List,
        }
    }
}

impl ParamSpecBuilder {
    /// Mark this parameter as required (no default value).
    pub fn required(self) -> ParamSpec {
        ParamSpec {
            name: self.name,
            param_type: self.param_type,
            required: true,
            default: None,
        }
    }

    /// Mark this parameter as optional (no default value).
    pub fn optional(self) -> ParamSpec {
        ParamSpec {
            name: self.name,
            param_type: self.param_type,
            required: false,
            default: None,
        }
    }

    /// Set a default number value.
    pub fn default_num(self, value: f64) -> ParamSpec {
        use crate::ast::nodes::{NumberVal, SourceLoc};
        ParamSpec {
            name: self.name,
            param_type: self.param_type,
            required: false,
            default: Some(ParamValue::Number(Box::new(NumberVal::new(
                value,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            )))),
        }
    }

    /// Set a default string value.
    pub fn default_str(self, value: impl Into<String>) -> ParamSpec {
        use crate::ast::nodes::{SourceLoc, StringVal};
        ParamSpec {
            name: self.name,
            param_type: self.param_type,
            required: false,
            default: Some(ParamValue::String(Box::new(StringVal::new(
                value.into(),
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            )))),
        }
    }

    /// Set a default boolean value.
    pub fn default_bool(self, value: bool) -> ParamSpec {
        use crate::ast::nodes::{BoolVal, SourceLoc};
        ParamSpec {
            name: self.name,
            param_type: self.param_type,
            required: false,
            default: Some(ParamValue::Bool(Box::new(BoolVal::new(
                value,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            )))),
        }
    }

    /// Set a default bareword value.
    pub fn default_bareword(self, value: impl Into<String>) -> ParamSpec {
        use crate::ast::nodes::{BarewordVal, SourceLoc};
        ParamSpec {
            name: self.name,
            param_type: self.param_type,
            required: false,
            default: Some(ParamValue::Bareword(Box::new(BarewordVal::new(
                value.into(),
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            )))),
        }
    }

    /// Set a default shape value (list of dimensions).
    pub fn default_shape(self, dims: Vec<usize>) -> ParamSpec {
        use crate::ast::nodes::{ShapeVal, SourceLoc};
        ParamSpec {
            name: self.name,
            param_type: self.param_type,
            required: false,
            default: Some(ParamValue::Shape(Box::new(ShapeVal::new(
                dims,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            )))),
        }
    }
}

// ---------------------------------------------------------------------------
// BlockDef trait
// ---------------------------------------------------------------------------

/// Trait that every block type must implement.
pub trait BlockDef: Send + Sync {
    /// The block type name, e.g. "Conv2d", "Linear".
    fn name(&self) -> &str;

    /// The parameter specifications for this block type.
    fn params(&self) -> &[ParamSpec];

    /// Infer output shapes given input shapes and the resolved parameters.
    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String>;

    /// Optional: total parameter count for this block.
    fn param_count(
        &self,
        _inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        None
    }

    /// Whether block height should scale with channel depth in SVG layout.
    /// Set to `false` for passthrough blocks (activations, merges, etc.).
    fn show_depth(&self) -> bool {
        true
    }

    /// Number of tensor inputs this block expects.
    ///
    /// Return `None` for a variable number of inputs (e.g. `Concat`).
    /// The default is 1, which covers the vast majority of blocks.
    fn num_inputs(&self) -> Option<usize> {
        Some(1)
    }

    /// Number of tensor outputs this block produces.
    ///
    /// Return `None` for a variable number of outputs (e.g. `Split`).
    /// The default is 1, which covers the vast majority of blocks.
    fn num_outputs(&self) -> Option<usize> {
        Some(1)
    }
}
