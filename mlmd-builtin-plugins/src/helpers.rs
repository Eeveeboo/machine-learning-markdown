// ---------------------------------------------------------------------------
// Shared helper utilities for builtin block plugins.
//
// Port of src/plugins/builtins/_helpers.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::ast::nodes::ParamValue;

// ---------------------------------------------------------------------------
// Param accessors
// ---------------------------------------------------------------------------

/// Safely get a numeric param value.
/// Returns `None` if the key is absent or the value is not a number.
pub fn get_num(params: &HashMap<String, ParamValue>, key: &str) -> Option<f64> {
    match params.get(key)? {
        ParamValue::Number(n) => Some(n.value),
        _ => None,
    }
}

/// Safely get a string param value.  Accepts both `"string"` and `"bareword"`
/// kinds.  Returns `None` if the key is absent or the value is neither.
pub fn get_str(params: &HashMap<String, ParamValue>, key: &str) -> Option<String> {
    match params.get(key)? {
        ParamValue::String(s) => Some(s.value.clone()),
        ParamValue::Bareword(b) => Some(b.value.clone()),
        _ => None,
    }
}

/// Get a required numeric param, returning an error if absent or wrong type.
pub fn require_num(params: &HashMap<String, ParamValue>, key: &str) -> Result<f64, String> {
    match params.get(key) {
        None => Err(format!("Missing required numeric param \"{key}\"")),
        Some(ParamValue::Number(n)) => Ok(n.value),
        Some(other) => Err(format!(
            "Expected number for param \"{key}\", got \"{}\"",
            other.type_name()
        )),
    }
}

/// Get a list of numbers from a `shape` or `list` param.
///
/// - For a `shape` param, returns its dims (converted to `f64`).
/// - For a `list` param, returns only the numeric items.
/// - Returns an empty `Vec` if the key is absent or the value does not match.
pub fn get_num_list(params: &HashMap<String, ParamValue>, key: &str) -> Vec<f64> {
    match params.get(key) {
        None => vec![],
        Some(ParamValue::Shape(s)) => s.dims.iter().map(|&d| d as f64).collect(),
        Some(ParamValue::List(l)) => l
            .items
            .iter()
            .filter_map(|item| {
                if let ParamValue::Number(n) = item {
                    Some(n.value)
                } else {
                    None
                }
            })
            .collect(),
        Some(_) => vec![],
    }
}

/// Get a required shape param, returning an error if absent or wrong type.
/// Returns the dimensions as `Vec<usize>` (the `Shape` type).
pub fn require_shape(
    params: &HashMap<String, ParamValue>,
    key: &str,
) -> Result<Vec<usize>, String> {
    match params.get(key) {
        None => Err(format!("Missing required shape param \"{key}\"")),
        Some(ParamValue::Shape(s)) => Ok(s.dims.clone()),
        Some(other) => Err(format!(
            "Expected shape for param \"{key}\", got \"{}\"",
            other.type_name()
        )),
    }
}

// ---------------------------------------------------------------------------
// Shape arithmetic
// ---------------------------------------------------------------------------

/// Compute the output size of a convolution dimension.
///
/// Formula:  floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
///
/// Matches PyTorch's Conv1d/Conv2d/Conv3d semantics.
pub fn conv_output_size(
    input: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    let numerator = (input as isize) + 2 * (padding as isize)
        - (dilation as isize) * ((kernel as isize) - 1)
        - 1;
    if numerator < 0 {
        0
    } else {
        (numerator as usize) / stride + 1
    }
}

/// Simplified convolution output size (dilation = 1).
///
/// Equivalent to `conv_output_size(…, 1)`.
pub fn conv_out(size: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    conv_output_size(size, kernel, padding, stride, 1)
}

/// Compute the output size of a transposed convolution dimension.
///
/// Formula:  (input - 1) * stride - 2 * padding + kernel + output_padding
///
/// Matches PyTorch's ConvTranspose2d with dilation=1.
pub fn conv_transpose_output_size(
    input: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    output_padding: usize,
) -> usize {
    let result = (input as isize - 1) * (stride as isize)
        - 2 * (padding as isize)
        + (kernel as isize)
        + (output_padding as isize);
    if result < 0 {
        0
    } else {
        result as usize
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlmd_core::ast::nodes::{BarewordVal, ListVal, NumberVal, ShapeVal, SourceLoc, StringVal};

    fn loc() -> SourceLoc {
        SourceLoc {
            line: 0,
            col: 0,
            offset: 0,
        }
    }

    fn sample_params() -> HashMap<String, ParamValue> {
        let mut m = HashMap::new();
        m.insert(
            "lr".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(0.001, loc()))),
        );
        m.insert(
            "name".to_string(),
            ParamValue::String(Box::new(StringVal::new("test".into(), loc()))),
        );
        m.insert(
            "act".to_string(),
            ParamValue::Bareword(Box::new(BarewordVal::new("relu".into(), loc()))),
        );
        m.insert(
            "dims".to_string(),
            ParamValue::Shape(Box::new(ShapeVal::new(vec![64, 128, 3], loc()))),
        );
        m.insert(
            "vals".to_string(),
            ParamValue::List(Box::new(ListVal::new(
                vec![
                    ParamValue::Number(Box::new(NumberVal::new(1.0, loc()))),
                    ParamValue::Number(Box::new(NumberVal::new(2.5, loc()))),
                    ParamValue::Number(Box::new(NumberVal::new(3.0, loc()))),
                ],
                loc(),
            ))),
        );
        m
    }

    // -- get_num -----------------------------------------------------------

    #[test]
    fn test_get_num_found() {
        let p = sample_params();
        assert!((get_num(&p, "lr").unwrap() - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_num_missing() {
        let p = sample_params();
        assert_eq!(get_num(&p, "nonexistent"), None);
    }

    #[test]
    fn test_get_num_wrong_type() {
        let p = sample_params();
        assert_eq!(get_num(&p, "name"), None); // string, not number
    }

    // -- get_str -----------------------------------------------------------

    #[test]
    fn test_get_str_string() {
        let p = sample_params();
        assert_eq!(get_str(&p, "name"), Some("test".to_string()));
    }

    #[test]
    fn test_get_str_bareword() {
        let p = sample_params();
        assert_eq!(get_str(&p, "act"), Some("relu".to_string()));
    }

    #[test]
    fn test_get_str_missing() {
        let p = sample_params();
        assert_eq!(get_str(&p, "nonexistent"), None);
    }

    #[test]
    fn test_get_str_wrong_type() {
        let p = sample_params();
        assert_eq!(get_str(&p, "lr"), None); // number, not string/bareword
    }

    // -- require_num -------------------------------------------------------

    #[test]
    fn test_require_num_ok() {
        let p = sample_params();
        assert!((require_num(&p, "lr").unwrap() - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_require_num_missing() {
        let p = sample_params();
        let err = require_num(&p, "nonexistent").unwrap_err();
        assert!(err.contains("nonexistent"));
    }

    #[test]
    fn test_require_num_wrong_type() {
        let p = sample_params();
        let err = require_num(&p, "name").unwrap_err();
        assert!(err.contains("name"));
        assert!(err.contains("Expected number"));
    }

    // -- get_num_list ------------------------------------------------------

    #[test]
    fn test_get_num_list_missing() {
        let p = sample_params();
        let r = get_num_list(&p, "nonexistent");
        assert!(r.is_empty());
    }

    #[test]
    fn test_get_num_list_shape() {
        let p = sample_params();
        let r = get_num_list(&p, "dims");
        assert_eq!(r, vec![64.0, 128.0, 3.0]);
    }

    #[test]
    fn test_get_num_list_list() {
        let p = sample_params();
        let r = get_num_list(&p, "vals");
        assert_eq!(r.len(), 3);
        assert!((r[0] - 1.0).abs() < f64::EPSILON);
        assert!((r[1] - 2.5).abs() < f64::EPSILON);
        assert!((r[2] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_num_list_wrong_type() {
        let p = sample_params();
        let r = get_num_list(&p, "name"); // string, not shape/list
        assert!(r.is_empty());
    }

    // -- require_shape -----------------------------------------------------

    #[test]
    fn test_require_shape_ok() {
        let p = sample_params();
        assert_eq!(require_shape(&p, "dims").unwrap(), vec![64, 128, 3]);
    }

    #[test]
    fn test_require_shape_missing() {
        let p = sample_params();
        let err = require_shape(&p, "nonexistent").unwrap_err();
        assert!(err.contains("nonexistent"));
    }

    #[test]
    fn test_require_shape_wrong_type() {
        let p = sample_params();
        let err = require_shape(&p, "lr").unwrap_err();
        assert!(err.contains("lr"));
        assert!(err.contains("Expected shape"));
    }

    // -- conv_output_size --------------------------------------------------

    #[test]
    fn test_conv_output_size_typical() {
        // input=7, kernel=3, padding=1, stride=2, dilation=1
        // ((7+2-2-1)/2)+1 = (6/2)+1 = 4
        assert_eq!(conv_output_size(7, 3, 1, 2, 1), 4);
    }

    #[test]
    fn test_conv_output_size_dilation() {
        // input=7, kernel=3, padding=2, stride=1, dilation=2
        // ((7+4-2*(2)-1)/1)+1 = ((7+4-4-1)/1)+1 = (6/1)+1 = 7
        assert_eq!(conv_output_size(7, 3, 2, 1, 2), 7);
    }

    #[test]
    fn test_conv_output_size_no_padding() {
        // input=5, kernel=3, padding=0, stride=1
        // ((5+0-1*(2)-1)/1)+1 = ((5-2-1)/1)+1 = (2/1)+1 = 3
        assert_eq!(conv_output_size(5, 3, 0, 1, 1), 3);
    }

    #[test]
    fn test_conv_output_size_negative_clamps() {
        // If the intermediate value goes negative, we clamp to 0.
        assert_eq!(conv_output_size(1, 5, 0, 1, 1), 0);
    }

    // -- conv_out ----------------------------------------------------------

    #[test]
    fn test_conv_out() {
        // Same as conv_output_size with dilation=1
        assert_eq!(conv_out(7, 3, 2, 1), conv_output_size(7, 3, 1, 2, 1));
        assert_eq!(conv_out(5, 3, 1, 0), 3);
    }

    // -- conv_transpose_output_size ----------------------------------------

    #[test]
    fn test_conv_transpose_output_size_typical() {
        // input=4, kernel=3, padding=1, stride=2, output_padding=0
        // (4-1)*2 - 2*1 + 3 + 0 = 6 - 2 + 3 = 7
        assert_eq!(conv_transpose_output_size(4, 3, 1, 2, 0), 7);
    }

    #[test]
    fn test_conv_transpose_output_size_no_padding() {
        // input=3, kernel=3, padding=0, stride=1, output_padding=0
        // (3-1)*1 - 0 + 3 + 0 = 2 + 3 = 5
        assert_eq!(conv_transpose_output_size(3, 3, 0, 1, 0), 5);
    }

    #[test]
    fn test_conv_transpose_output_size_with_output_padding() {
        // input=4, kernel=3, padding=1, stride=2, output_padding=1
        // (4-1)*2 - 2*1 + 3 + 1 = 6 - 2 + 3 + 1 = 8
        assert_eq!(conv_transpose_output_size(4, 3, 1, 2, 1), 8);
    }
}
