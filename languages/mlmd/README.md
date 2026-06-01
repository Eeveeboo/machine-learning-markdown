# MLMD Legacy Zed Language Config

This directory provides basic language server support for `.mlmd` files when installed via:

```bash
mlmd install zed
```

This copies the config to `~/.config/zed/languages/mlmd/`.

## Limitations

This legacy approach does NOT provide syntax highlighting (tree-sitter grammar not included).

## For Full Zed Support

Use the [zed-mlmd dev extension](../../zed-mlmd/) instead, which provides:
- Syntax highlighting
- Bracket matching
- Outline panel
- Language server integration
