pub mod completions;
pub mod definition;
pub mod diagnostics;
pub mod hover;
pub mod references;
pub mod server;

// ---------------------------------------------------------------------------
// Shared UTF-16 ↔ UTF-8 offset conversion
// ---------------------------------------------------------------------------

/// Convert an LSP character offset (UTF-16 code units) to a byte offset into a Rust `&str`.
///
/// LSP [`Position::character`] is defined as UTF-16 code units (1 for BMP chars,
/// 2 for surrogate pairs). Rust `&str` is UTF-8. This function maps between them.
///
/// Returns `None` if `utf16_offset` is beyond the string's character boundaries.
pub fn utf16_offset_to_byte_index(text: &str, utf16_offset: u32) -> Option<usize> {
    if utf16_offset == 0 {
        return Some(0);
    }
    let mut utf16_count: u32 = 0;
    for (byte_idx, ch) in text.char_indices() {
        let units = if ch as u32 >= 0x10000 { 2 } else { 1 };
        if utf16_count + units > utf16_offset {
            return Some(byte_idx);
        }
        utf16_count += units;
    }
    // If offset points exactly past the last character (end of string)
    if utf16_count == utf16_offset {
        Some(text.len())
    } else {
        None
    }
}

fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Extract the word (identifier) under the given LSP character position.
///
/// `character` is an LSP UTF-16 code-unit offset.
/// The function converts it to a byte index, confirms we're on a word-character,
/// then expands left and right to capture the full identifier.
pub fn word_at_position(line: &str, character: u32) -> Option<String> {
    let byte_idx = utf16_offset_to_byte_index(line, character)?;
    if byte_idx >= line.len() {
        return None;
    }

    let bytes = line.as_bytes();
    // Check if we're on a word character
    if !bytes[byte_idx].is_ascii_alphanumeric() && bytes[byte_idx] != b'_' {
        return None;
    }

    // Expand backward to start of word
    let mut start = byte_idx;
    while start > 0 {
        if is_word_char(bytes[start - 1]) {
            start -= 1;
        } else {
            break;
        }
    }

    // Expand forward to end of word
    let mut end = byte_idx + 1;
    while end < bytes.len() {
        if is_word_char(bytes[end]) {
            end += 1;
        } else {
            break;
        }
    }

    Some(line[start..end].to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf16_offset_ascii() {
        let s = "hello";
        assert_eq!(utf16_offset_to_byte_index(s, 0), Some(0));
        assert_eq!(utf16_offset_to_byte_index(s, 1), Some(1));
        assert_eq!(utf16_offset_to_byte_index(s, 4), Some(4));
        assert_eq!(utf16_offset_to_byte_index(s, 5), Some(5));
        assert_eq!(utf16_offset_to_byte_index(s, 6), None);
    }

    #[test]
    fn test_utf16_offset_multibyte_utf8() {
        // '€' (U+20AC) is 3 UTF-8 bytes but 1 UTF-16 unit
        let s = "a€z";
        // char_indices: (0,'a',1unit), (1,'€',1unit), (4,'z',1unit)
        assert_eq!(utf16_offset_to_byte_index(s, 0), Some(0)); // before 'a'
        assert_eq!(utf16_offset_to_byte_index(s, 1), Some(1)); // after 'a' → byte 1
        assert_eq!(utf16_offset_to_byte_index(s, 2), Some(4)); // after '€' → byte 4
        assert_eq!(utf16_offset_to_byte_index(s, 3), Some(5)); // after 'z' → byte 5 (end)
        assert_eq!(utf16_offset_to_byte_index(s, 4), None);
    }

    #[test]
    fn test_utf16_offset_chinese() {
        // Chinese chars U+4E00..U+9FFF: 3 UTF-8 bytes, 1 UTF-16 unit each
        let s = "你好"; // 2 chars
        assert_eq!(utf16_offset_to_byte_index(s, 0), Some(0));
        assert_eq!(utf16_offset_to_byte_index(s, 1), Some(3)); // after 你
        assert_eq!(utf16_offset_to_byte_index(s, 2), Some(6)); // after 好 (end)
        assert_eq!(utf16_offset_to_byte_index(s, 3), None);
    }

    #[test]
    fn test_utf16_offset_surrogate_pair() {
        // 😀 (U+1F600) = 4 UTF-8 bytes, 2 UTF-16 units (surrogate pair)
        let s = "a😀z";
        // char_indices: (0,'a',1), (1,'😀',2), (5,'z',1)
        assert_eq!(utf16_offset_to_byte_index(s, 0), Some(0)); // before 'a'
        assert_eq!(utf16_offset_to_byte_index(s, 1), Some(1)); // after 'a'
        assert_eq!(utf16_offset_to_byte_index(s, 2), Some(1)); // middle of surrogate pair → snaps to emoji start
        assert_eq!(utf16_offset_to_byte_index(s, 3), Some(5)); // after emoji → byte 5
        assert_eq!(utf16_offset_to_byte_index(s, 4), Some(6)); // after 'z' → byte 6 (end)
        assert_eq!(utf16_offset_to_byte_index(s, 5), None);
    }

    #[test]
    fn test_utf16_offset_empty() {
        assert_eq!(utf16_offset_to_byte_index("", 0), Some(0));
        assert_eq!(utf16_offset_to_byte_index("", 1), None);
    }

    #[test]
    fn test_word_at_position_ascii() {
        assert_eq!(
            word_at_position("hello world", 0),
            Some("hello".to_string())
        );
        assert_eq!(
            word_at_position("hello world", 6),
            Some("world".to_string())
        );
        assert_eq!(word_at_position("hello world", 5), None); // space
    }

    #[test]
    fn test_word_at_position_non_ascii_line() {
        // The line has non-ASCII chars, but the word itself is ASCII
        let line = "a € bcd";
        assert_eq!(word_at_position(line, 5), Some("bcd".to_string()));
        // character 5 in UTF-16: a(0), space(1), €(2), space(3), b(4), c(5), d(6)
        // byte 5 is 'c' (actually let me trace: a=byte0, space=byte1, €=bytes2-4, space=byte5, b=byte6...)
        // Wait, "a € bcd" has: 'a'(0), ' '(1), '€'(2,3,4), ' '(5), 'b'(6), 'c'(7), 'd'(8)
        // UTF-16 offsets: 0:a, 1:space, 2:€, 3:space, 4:b, 5:c, 6:d
        // So offset 5 = 'c' in UTF-16 = byte 7, word "bcd"
        assert_eq!(word_at_position(line, 4), Some("bcd".to_string())); // at 'b'
        assert_eq!(word_at_position(line, 6), Some("bcd".to_string())); // at 'd'
    }

    #[test]
    fn test_word_at_position_underscore() {
        assert_eq!(
            word_at_position("foo_bar baz", 0),
            Some("foo_bar".to_string())
        );
        assert_eq!(
            word_at_position("foo_bar baz", 4),
            Some("foo_bar".to_string()) // at '_'
        );
    }

    #[test]
    fn test_word_at_position_out_of_bounds() {
        assert_eq!(word_at_position("hi", 10), None);
    }

    #[test]
    fn test_word_at_position_empty() {
        assert_eq!(word_at_position("", 0), None);
    }
}
