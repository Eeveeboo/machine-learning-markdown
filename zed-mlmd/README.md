# zed-mlmd

Zed editor extension for the [MLMD](https://github.com/eeveeboo/nnml) neural network DSL.

## Features

- Syntax highlighting for `.mlmd` files
- Bracket matching
- Outline panel support
- Language server (diagnostics, hover, completions)

## Development Installation

1. **Generate the parser** (requires `tree-sitter-cli`):
   ```bash
   npm install -g tree-sitter-cli   # one-time global install
   cd zed-mlmd/grammars/mlmd && npm install
   tree-sitter generate
   cd ../..
   ```
   > **Note:** You do **not** need to compile WASM manually. When installing as a
   > dev extension, Zed compiles the grammar from the generated C source automatically.

2. **Install as dev extension in Zed**:
   - Open Zed
   - Press `Cmd+Shift+X` → click "Install Dev Extension"
   - Select the repo root (**`<repo>/nnml/`**, not `zed-mlmd/`) — this is where
     `extension.toml` now lives

3. Open any `.mlmd` file — syntax highlighting and LSP should activate.

## Building the Rust extension manually

```bash
cd zed-mlmd
./build.sh
```

This generates the grammar, builds the Rust WASM component, and copies
`extension.wasm` to the repo root (where Zed expects it alongside `extension.toml`).

## Requirements

- [Zed editor](https://zed.dev)
- `tree-sitter-cli` for generating the parser (`npm install -g tree-sitter-cli`)
- Node.js for the MLMD language server (called via `node dist/bin/mlmd.js lsp`)
