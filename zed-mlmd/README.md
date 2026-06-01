# zed-mlmd — Zed Editor Extension

Zed extension source for the [MLMD](https://github.com/eeveeboo/nnml) neural network DSL.

## Features

- Syntax highlighting (tree-sitter grammar)
- Bracket matching
- Outline panel support
- LSP integration (diagnostics, hover, completions)

## Quick Start

Run from the repo root:

```bash
mlmd install zed
```

This generates the grammar parser, builds the grammar WASM, compiles the Rust
extension as a WASM component, and configures `extension.toml` for your machine.

Then open Zed → **Cmd+Shift+X** → "Install Dev Extension" → select the repo root.

## Manual Build

```bash
cd zed-mlmd
./build.sh
```

## Development Notes

- The tree-sitter grammar source is at `../grammars/mlmd-grammar/` (not in this directory)
- `extension.toml` lives at the repo root (`../extension.toml`)
- The LSP server is called via `worktree.which("mlmd")` (global install) or
  `worktree.root_path()/dist/bin/mlmd.js` (dev mode)

## Requirements

- [Zed editor](https://zed.dev)
- Rust `wasm32-wasip2` target for building the extension WASM
- Node.js for the MLMD language server
