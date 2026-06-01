# zed-mlmd

Zed editor extension for the [MLMD](https://github.com/eeveeboo/nnml) neural network DSL.

## Features

- Syntax highlighting for `.mlmd` files
- Bracket matching
- Outline panel support
- Language server (diagnostics, hover, completions) via `npx mlmd lsp`

## Development Installation

1. **Generate the parser** (requires `tree-sitter-cli`):
   ```bash
   npm install -g tree-sitter-cli   # one-time global install
   cd grammars/mlmd && npm install
   tree-sitter generate
   cd ../..
   ```
   > **Note:** You do **not** need to compile WASM manually. When installing as a
   > dev extension, Zed compiles the grammar from the generated C source automatically.

2. **Install as dev extension in Zed**:
   - Open Zed
   - Press `Cmd+Shift+X` → click "Install Dev Extension"
   - Select this `zed-mlmd/` directory

3. Open any `.mlmd` file — syntax highlighting and LSP should activate.

## Building WASM manually (optional)

If you want to test the grammar outside of Zed (e.g. with `tree-sitter parse`),
you can build the WASM bundle with `./build.sh`. This requires either
[emscripten](https://emscripten.org/docs/getting_started/downloads.html) or
Docker to be available.

## Requirements

- [Zed editor](https://zed.dev)
- `tree-sitter-cli` for generating the parser (`npm install -g tree-sitter-cli`)
- `mlmd` CLI on PATH (or accessible via `npx`) for LSP features
