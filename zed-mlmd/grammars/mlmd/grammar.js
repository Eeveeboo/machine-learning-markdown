module.exports = grammar({
  name: 'mlmd',

  extras: $ => [
    /\s+/,
    $.comment,
  ],

  conflicts: $ => [
    [$.tensor_ref, $.list_value],
  ],

  rules: {
    source_file: $ => repeat(
      choice(
        $.group_declaration,
        $.chain,
      )
    ),

    // # comment text
    comment: $ => token(seq('#', /.*/)),

    // [[ Name ]] or [[ Name > Sub ]]
    group_declaration: $ => seq(
      $.group_open,
      $._group_name,
      repeat(seq('>', $._group_name)),
      $.group_close,
    ),

    _group_name: $ => choice($.identifier, $.block_type),
    group_open: $ => '[[',
    group_close: $ => ']]',

    // A chain: elements connected by arrows
    // Can start with tensor_ref or block
    chain: $ => choice(
      // tensor_ref -> ... (join)
      seq($.tensor_ref, $.arrow, $.block, repeat(seq($.arrow, $._chain_tail))),
      // block -> ... OR block alone
      seq($.block, repeat(seq($.arrow, $._chain_tail))),
    ),

    _chain_tail: $ => choice(
      $.tensor_ref,  // fork: -> [name1, name2]
      $.block,
    ),

    // Block: PascalCase type + optional params
    block: $ => seq(
      $.block_type,
      optional($.params),
    ),

    block_type: $ => /[A-Z][a-zA-Z0-9_]*/,

    // [ name ] or [ name1, name2 ]  — tensor reference at chain level
    tensor_ref: $ => seq(
      '[',
      $.identifier,
      repeat(seq(',', $.identifier)),
      ']',
    ),

    // ( param=value, ... )
    params: $ => seq(
      $.params_open,
      optional(seq($.param, repeat(seq(',', $.param)))),
      $.params_close,
    ),
    params_open: $ => '(',
    params_close: $ => ')',

    param: $ => seq(
      $.param_key,
      '=',
      $._value,
    ),

    param_key: $ => /[a-z_][a-zA-Z0-9_]*/,

    _value: $ => choice(
      $.number,
      $.string,
      $.boolean,
      $.shape_value,   // (1,28,28)
      $.list_value,    // [64,128,256]
      $.identifier,    // bareword value
      $.block_type,    // PascalCase bareword (e.g. mode=ReLU)
    ),

    // Shape: (1, 28, 28) — tuple of numbers
    shape_value: $ => seq(
      '(',
      $.number,
      repeat(seq(',', $.number)),
      optional(','),
      ')',
    ),

    // List: [64, 128, 256]
    list_value: $ => seq(
      '[',
      $._value,
      repeat(seq(',', $._value)),
      ']',
    ),

    number: $ => token(
      seq(
        optional('-'),
        /\d+/,
        optional(seq('.', /\d+/)),
        optional(seq(/[eE]/, optional(/[+-]/), /\d+/)),
      )
    ),

    string: $ => seq(
      '"',
      repeat(choice(
        /[^"\\]+/,
        /\\./,
      )),
      '"',
    ),

    boolean: $ => choice('true', 'false'),

    arrow: $ => '->',

    identifier: $ => /[a-z_][a-zA-Z0-9_]*/,
  },
});
