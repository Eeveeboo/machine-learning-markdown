# MLMD — Neural Network Architecture DSL

[![CI](https://github.com/Eeveeboo/machine-learning-markdown/actions/workflows/ci.yml/badge.svg)](https://github.com/Eeveeboo/machine-learning-markdown/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

MLMD is a DSL (domain-specific language) for describing neural network architectures in a concise, readable format. It compiles `.mlmd` files into multiple representations:

- **SVG diagrams** — standalone architecture visualization
- **PyTorch code** — runnable `nn.Module` subclasses
- **Keras code** — functional API models
- **Candle (Rust) code** — struct + impl for the [Candle](https://github.com/huggingface/candle) framework

The toolchain includes a CLI, a Rust API, a tree-sitter grammar, an LSP server, and editor extensions for VS Code and Zed.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Language Guide](#language-guide)
- [Block Reference](#block-reference)
- [Examples](#examples)
- [Editor Support](#editor-support)
- [Plugin System](#plugin-system)
- [LSP Features](#lsp-features)
- [Rust API](#rust-api)
- [Development](#development)

---

## Quick Start

Create a file `unet.mlmd`:

```mlmd
# U-Net encoder-decoder with skip connections
# Input: 1x64x64 (single-channel image)

[[ UNet ]]

[[ UNet > Encoder ]]

# Encoder level 1: produces skip connection and continues downward
Input(shape=(1,64,64))
    -> Conv2d(filters=32, kernel=3, padding=1) -> ReLU -> [enc1_skip, enc1_down]

[[ UNet > Bottleneck ]]

# Encoder level 2 + bottleneck
[enc1_down] -> MaxPool(kernel=2, stride=2)
            -> Conv2d(filters=64, kernel=3, padding=1) -> ReLU -> [enc2_skip, enc2_down]

[enc2_down]  -> MaxPool(kernel=2, stride=2)
             -> Conv2d(filters=128, kernel=3, padding=1) -> ReLU -> [bottleneck]

[[ UNet > Decoder ]]

# Decoder level 1: upsample bottleneck, merge with enc2 skip
[bottleneck] -> TransposedConv2d(filters=64, kernel=2, stride=2) -> [up1]
[up1, enc2_skip] -> Concat(axis=0)
                 -> Conv2d(filters=64, kernel=3, padding=1) -> ReLU -> [dec1]

# Decoder level 2: upsample, merge with enc1 skip
[dec1]           -> TransposedConv2d(filters=32, kernel=2, stride=2) -> [up2]
[up2, enc1_skip] -> Concat(axis=0)
                 -> Conv2d(filters=32, kernel=3, padding=1) -> ReLU
                 -> Conv2d(filters=1, kernel=1)
                 -> Output
```

| Renders as                |
|---------------------------|
| ![](./examples/unet.svg)  |

From the repository root, build and run:

```bash
# Build the CLI
cargo build --release -p mlmd

# Run directly with cargo
cargo run -p mlmd -- visualize examples/lenet.mlmd -o lenet.svg

# Or use the built binary
./target/release/mlmd visualize examples/lenet.mlmd -o lenet.svg
```

Alternatively, use the `Makefile`:

```bash
make build
make generate-examples
make test
```

---

## Installation

### Prerequisites

- **Rust toolchain** (stable) — install via [rustup](https://rustup.rs/)
- **Tree-sitter CLI** (optional, for grammar development): `npm install -g tree-sitter-cli`

### Clone and Build

```bash
git clone https://github.com/Eeveeboo/machine-learning-markdown.git
cd machine-learning-markdown
cargo build --workspace
```

This builds all crates including the `mlmd` CLI binary at `target/debug/mlmd`.

### Install Globally

```bash
cargo install --path mlmd
mlmd --help
```

Or symlink the release binary:

```bash
cargo build --release -p mlmd
ln -s "$(pwd)/target/release/mlmd" /usr/local/bin/mlmd
```

---

## CLI Reference

### `mlmd --help`

```
Usage: mlmd [options] [command]

CLI for describing and validating neural network architectures.

Options:
  -V, --version          output the version number
  -h, --help             display help for command

Commands:
  visualize [options] <file>    Visualize a .mlmd model architecture as SVG
  generate [options] <file>     Generate code from a .mlmd model
  lint [options] <file>         Lint a .mlmd file for errors and warnings
  lsp [options]                 Start the Language Server Protocol server
  install [options] <command>   Install MLMD editor extensions
  plugin [options] <command>    Manage external plugins (list, install, info)
  help [command]                display help for command
```

### `mlmd visualize <file>`

Generate an SVG diagram of the architecture.

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output <file>` | Output SVG file path | stdout |
| `-f, --format <format>` | Output format | `svg` |

```bash
mlmd visualize model.mlmd -o model.svg
mlmd visualize model.mlmd                          # prints SVG to stdout
```

### `mlmd generate <file>`

Generate framework code from a `.mlmd` model file.

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output <dir>` | Output directory | `.` (or `targets[].out` from `.mlmdrc`) |
| `-t, --target <target>` | Target framework | `pytorch` (or first target from `.mlmdrc`) |

**Targets:** `pytorch`, `keras`, `candle`

```bash
mlmd generate model.mlmd -t pytorch -o ./generated
mlmd generate model.mlmd -t candle -o ./generated
mlmd generate model.mlmd                           # uses defaults from .mlmdrc or pytorch
```

### `mlmd lint <file>`

Lint a `.mlmd` file for errors (shape mismatches, cycles, missing parameters, undefined references).

| Option | Description | Default |
|--------|-------------|---------|
| `--json` | Output diagnostics as JSON | human-readable text |

```bash
mlmd lint model.mlmd
mlmd lint model.mlmd --json                        # machine-readable output
```

### `mlmd lsp`

Start the LSP server on stdin/stdout. Used by editor extensions for diagnostics, hover, autocomplete, and go-to-definition.

```bash
mlmd lsp --stdio
```

### `mlmd install <command>`

Install editor extensions (see [Editor Support](#editor-support)).

```bash
mlmd install vscode                                # install VS Code extension
mlmd install zed                                   # prepare Zed dev extension
```

### `mlmd plugin <command>`

Manage external plugins.

```bash
mlmd plugin list                                   # list installed plugins
mlmd plugin install <path>                         # install a plugin from path
mlmd plugin info <name>                            # show plugin details
```

---

## Configuration

Create a `.mlmdrc` JSON file in your project root for shared defaults:

```json
{
  "plugins": "./mlmd-plugins",
  "targets": [
    { "lang": "pytorch", "out": "./generated" },
    { "lang": "keras", "out": "./generated" },
    { "lang": "candle", "out": "./generated" }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `plugins` | `string` | Path to a directory of custom block plugins (WASM components) |
| `targets` | `array` | List of default code generation targets |

When no `-t`/`--target` is passed, `mlmd generate` uses the targets from `.mlmdrc`.

---

## Language Guide

### Syntax Overview

| Construct | Example |
|-----------|---------|
| **Block declaration** | `Conv2d(filters=64, kernel=3, padding=1)` |
| **Sequential chain** | `Block1 -> Block2 -> Block3` |
| **Tensor naming (fork)** | `Block -> [tensor_name]` |
| **Tensor joining** | `[tensor_a, tensor_b] -> MergeBlock` |
| **Groups** | `[[ GroupPath > Component ]]` |
| **Comments** | `# single-line comment` |
| **Multi-line chains** | Newline after `->` continues the chain |
| **Multi-line params** | Params span multiple lines inside parentheses |

### Architecture Flow

A `.mlmd` file describes a dataflow graph:

```
Input(shape=(...)) -> [layer after layer] -> Output
```

- Every chain starts from `Input` and ends at `Output`.
- Use `[name]` to name a tensor (fork point).
- Use `[name1, name2]` to join multiple tensors into a merge block.

### Groups

Groups organize architectures hierarchically and map to `nn.Module` subclasses in PyTorch:

```mlmd
[[ Encoder > Block ]]
Input(shape=(3,224,224))
    -> Conv2d(filters=64, kernel=3, padding=1)
    -> [encoder_output]

[[ Decoder > Block ]]
[encoder_output] -> ConvTransposed2d(filters=3, kernel=3)
    -> Output
```

Groups can be nested with `>` separators: `[[ Parent > Child > Grandchild ]]`.

---

## Block Reference (Builtin)

### Layers

| Block | Parameters |
|-------|------------|
| `Input` | `shape=...` |
| `Output` | _(none)_ |
| `Linear` | `in_features`, `out_features`, `bias`(bool) |
| `Embedding` | `num_embeddings`, `embedding_dim` |
| `Conv1d` / `Conv2d` / `Conv3d` | `filters`, `kernel`, `stride`, `padding`, `dilation`, `groups`, `bias`(bool) |
| `TransposedConv2d` | `filters`, `kernel`, `stride`, `padding`, `output_padding`, `bias`(bool) |

### Activations

| Block | Parameters |
|-------|------------|
| `ReLU` | `inplace`(bool) |
| `LeakyReLU` | `negative_slope` |
| `PReLU` | `num_parameters` |
| `ELU` | `alpha` |
| `SELU` | _(none)_ |
| `GELU` | _(none)_ |
| `SiLU` | _(none)_ |
| `Sigmoid` | _(none)_ |
| `Tanh` | _(none)_ |
| `Softmax` | `dim` |

### Normalization

| Block | Parameters |
|-------|------------|
| `BatchNorm` | `features`, `eps`, `momentum`, `affine`(bool) |
| `LayerNorm` | `normalized_shape`, `eps`, `affine`(bool) |
| `GroupNorm` | `num_groups`, `num_channels`, `eps`, `affine`(bool) |
| `InstanceNorm` | `features`, `eps`, `affine`(bool) |

### Pooling

| Block | Parameters |
|-------|------------|
| `MaxPool` | `kernel`, `stride`, `padding`, `dilation`, `ceil_mode`(bool) |
| `AvgPool` | `kernel`, `stride`, `padding`, `ceil_mode`(bool) |
| `GlobalAvgPool` | _(none)_ |
| `AdaptiveAvgPool` | `output_size` |

### Recurrent

| Block | Parameters |
|-------|------------|
| `LSTM` | `input_size`, `hidden_size`, `num_layers`, `bidirectional`(bool), `bias`(bool) |
| `GRU` | `input_size`, `hidden_size`, `num_layers`, `bidirectional`(bool), `bias`(bool) |
| `RNN` | `input_size`, `hidden_size`, `num_layers`, `nonlinearity`, `bidirectional`(bool), `bias`(bool) |

### Transform

| Block | Parameters |
|-------|------------|
| `Flatten` | `start_dim`, `end_dim` |
| `Reshape` | `shape` |
| `Dropout` | `p`, `inplace`(bool) |
| `Pad` | `padding`, `mode` |

### Merge / Arithmetic

| Block | Parameters |
|-------|------------|
| `Add`, `Mul`, `Sub`, `Div` | _(none)_ |
| `Concat` | `axis` |
| `MatMul` | `transpose_a`(bool), `transpose_b`(bool) |

### Control Flow

| Block | Parameters |
|-------|------------|
| `Split` | `sizes` (list of ints), `axis` |
| `Repeat` | `n` |
| `Map` | `block` (block type reference) |
| `Gather` | `axis`, `index` |

---

## Examples

The `examples/` directory contains five complete architectures with generated outputs:

| File | Architecture | Highlights |
|------|-------------|------------|
| [`lenet.mlmd`](examples/lenet.mlmd) | LeNet-5 | Sequential conv → pool → linear pipeline |
| [`resnet-bottleneck.mlmd`](examples/resnet-bottleneck.mlmd) | ResNet Bottleneck | Skip connection with fork/join |
| [`attention.mlmd`](examples/attention.mlmd) | Single-Head Attention | Multi-input: Q, K, V branches |
| [`inception.mlmd`](examples/inception.mlmd) | Inception Module | 4 parallel branches + Concat |
| [`unet.mlmd`](examples/unet.mlmd) | U-Net | Encoder → bottleneck → decoder with skips |

Each `.mlmd` file has corresponding generated outputs:

```
examples/
├── lenet.mlmd           # source
├── lenet.svg            # visualization
├── LeNet5.py            # PyTorch
├── LeNet5.rs            # Candle (Rust)
├── ...                  # (same pattern for all 5 architectures)
```

To regenerate all examples:

```bash
make generate-examples
```

---

## Editor Support

### VS Code

**Option A — Automatic install:**

```bash
# Build the CLI first
cargo build --release -p mlmd
./target/release/mlmd install vscode
```

### Zed

The `mlmd install zed` command sets up a full dev extension from your local checkout:

```bash
# Build the CLI first
cargo build --release -p mlmd
./target/release/mlmd install zed
```

This performs the full build pipeline:
- Generates the tree-sitter C parser from `grammar.js`
- Compiles grammar WASM (if tree-sitter WASI deps available; otherwise Zed compiles from source)
- Updates `extension.toml` with the local `file://` URL to the grammar repo
- Builds the Rust extension as a WASM component (`cargo build --target wasm32-wasip2`)
- Copies the extension WASM to the extension root

After running, install the dev extension in Zed:
1. Open Zed → press **Cmd+Shift+X** (or **Ctrl+Shift+X** on Linux)
2. Click **Install Dev Extension** (top-right)
3. Select this directory (`<project-root>`)
4. Open any `.mlmd` file — highlighting + LSP activate

**Features:**
- Syntax highlighting (tree-sitter grammar)
- Bracket matching
- Outline panel
- LSP integration

### Other Editors

MLMD ships with a standard TextMate grammar at [`syntaxes/mlmd.tmLanguage.json`](syntaxes/mlmd.tmLanguage.json), compatible with any editor that supports TextMate grammars (Sublime Text, TextMate, Nova, etc.).

The LSP server (`mlmd lsp`) communicates over stdin/stdout and can be integrated with any editor that supports the Language Server Protocol.

---

## Plugin System

Create custom block types as Rust libraries or WASM components.

### Writing a Plugin (Rust)

Plugins implement the `BlockPlugin` trait from `mlmd_core::plugin`:

```rust
use mlmd_core::plugin::{BlockPlugin, PluginRegistry};
use mlmd_core::types::{Shape, ParamValue};

struct MyCustomBlock;

impl BlockPlugin for MyCustomBlock {
    fn name(&self) -> &str { "MyCustomBlock" }
    fn inputs(&self) -> &[&str] { &["x"] }
    fn outputs(&self) -> &[&str] { &["y"] }
    fn params(&self) -> Vec<(&str, ParamValue)> {
        vec![("alpha", ParamValue::Number(0.5))]
    }
    fn infer_shape(&self, inputs: &[Shape], _params: &[ParamValue]) -> Vec<Shape> {
        vec![inputs[0].clone()]
    }
}
```

### Plugin Discovery

The CLI loads `.wasm` / `.so` / `.dylib` files from the plugins directory specified in `.mlmdrc`.

### Usage

1. Create a plugins directory (e.g., `./mlmd-plugins/`).
2. Point to it in `.mlmdrc`:
   ```json
   { "plugins": "./mlmd-plugins" }
   ```
3. Use the custom block in `.mlmd` files:
   ```mlmd
   MyCustomBlock(alpha=0.3)
   ```

Shape inference, SVG rendering, and code generation all use your plugin.

---

## LSP Features

The MLMD language server provides:

| Feature | Description |
|---------|-------------|
| **Diagnostics** | Real-time lint errors (shape mismatches, cycles, missing params) |
| **Hover** | Type information on hover over blocks and tensors |
| **Autocomplete** | Block type names, parameter names, tensor references |
| **Go-to-Definition** | Jump to block type definitions |
| **Find References** | Locate all tensor usages |

Start the server:

```bash
mlmd lsp --stdio
```

---

## Rust API

Use `mlmd_core` as a library in your Rust project:

```toml
[dependencies]
mlmd-core = { path = "../mlmd-core" }
mlmd-builtin-plugins = { path = "../mlmd-builtin-plugins" }
```

```rust
use mlmd_core::parser::{tokenize, parse};
use mlmd_core::ast::graph::build_graph;
use mlmd_core::shape::infer::infer_shapes;
use mlmd_core::codegen::target::get_target;
use mlmd_core::visualize::{layout, render_svg};
use mlmd_core::lint::lint;
use mlmd_core::config::load_config;
use mlmd_core::plugin::registry::Registry;

// Register builtin blocks
mlmd_builtin_plugins::register_all();

// Register codegen targets
mlmd_core::codegen::targets::register_all_codegen_targets();

// Full pipeline
let tokens = tokenize(source)?;
let ast = parse(tokens)?;
let graph = build_graph(ast.nodes, &ast.groups)?;
let shaped = infer_shapes(&graph, &Registry::global())?;

// Code generation
let target = get_target("pytorch")?;
let files = target.generate(&shaped, &Registry::global());

// Visualization
let layout_result = layout(&shaped);
let svg = render_svg(&shaped, &layout_result);

// Linting
let diagnostics = lint(&graph, &Registry::global());
```

---

## Development

### Setup

```bash
git clone https://github.com/Eeveeboo/machine-learning-markdown.git
cd machine-learning-markdown
```

### Cargo Workspace

The project is organized as a Cargo workspace with these crates:

```
├── mlmd/                   # CLI binary entrypoint
├── mlmd-core/              # Core library (parser, AST, codegen, config, LSP, etc.)
├── mlmd-builtin-plugins/   # Built-in block plugin implementations (43 blocks)
├── mlmd-examples/          # Example generation + validation tests
├── mlmd-extension-zed/     # Zed editor extension (WASM component)
└── tests/e2e/rust/         # End-to-end tests (Candle model compilation)
```

### Scripts (via Makefile)

| Command | Description |
|---------|-------------|
| `make build` | Build all workspace crates (`cargo build --workspace`) |
| `make test` | Run all tests (`cargo test --workspace`) |
| `make lint` | Lint with clippy (`cargo clippy --workspace`) |
| `make fmt` | Check formatting (`cargo fmt --check`) |
| `make check` | Type-check without codegen (`cargo check --workspace`) |
| `make generate-examples` | Regenerate all example outputs |
| `make clean` | Clean build artifacts (`cargo clean`) |

### Testing

```bash
cargo test --workspace              # run all tests
cargo test -p mlmd-core             # test core library only
cargo test -p mlmd-builtin-plugins  # test built-in blocks
cargo test -p mlmd-examples         # regenerate + validate examples
cargo test                          # run tests for the root `mlmd` binary
cargo test -- --test-threads=1      # sequential (useful for shared resources)
```

### Running the CLI without Installing

```bash
cargo run -p mlmd -- visualize examples/lenet.mlmd -o lenet.svg
cargo run -p mlmd -- generate examples/lenet.mlmd -t pytorch -o ./out
cargo run -p mlmd -- lint examples/lenet.mlmd
cargo run -p mlmd -- lsp
```

### Building the Tree-Sitter Grammar

```bash
cd grammars/mlmd-grammar
npm install
npm exec tree-sitter generate
```

### Regenerating Example Outputs

```bash
make generate-examples
```

Or directly:

```bash
cargo run -p mlmd -- generate examples/lenet.mlmd -t pytorch -o examples
cargo run -p mlmd -- generate examples/lenet.mlmd -t candle -o examples
cargo run -p mlmd -- visualize examples/lenet.mlmd -o examples/lenet.svg
# ... repeat for all example files
```

---

