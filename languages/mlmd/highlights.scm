; MLMD Tree-sitter highlights query
; Maps syntax nodes to standard TextMate scope names

; Comments
(comment) @comment

; Block type keywords (PascalCase identifiers at block-head position)
; e.g. Network, Layer, Group
(block_type) @type

; Block names / identifiers
(identifier) @variable

; Group delimiters  [[  ]]
(group_open) @punctuation.bracket
(group_close) @punctuation.bracket

; Parameter parentheses  (  )
(params_open) @punctuation.bracket
(params_close) @punctuation.bracket

; Arrow operators  ->  =>
(arrow) @operator

; String literals
(string) @string

; Numeric literals
(number) @number

; Boolean literals
(boolean) @constant.builtin

; Parameter keys
(param_key) @property

; Punctuation
"," @punctuation.delimiter
"=" @operator
":" @punctuation.delimiter
