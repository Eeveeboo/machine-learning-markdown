# MLMD — Neural Network Architecture DSL

MLMD is a TypeScript CLI tool for describing neural network architectures using a concise DSL. It compiles `.mlmd` files to:
- **SVG diagrams** — standalone visualization of your architecture
- **PyTorch code** — runnable `nn.Module` subclasses
- **Keras code** — functional API models
- **Candle (Rust) code** — struct + impl for Candle framework

## Installation

```bash
npm install -g mlmd
```

Or run directly with npx:
```bash
npx mlmd --help
```

## Quick Start

Create a file `lenet.mlmd`:
```mlmd
# Inception module (GoogLeNet-style)
# Four parallel branches concatenated along channel axis
# Input: 192x28x28

[[ Inception > Module ]]

Input(shape=(192,28,28)) -> [b1, b2, b3, b4]

# Branch 1: 1x1 conv
[b1] -> Conv2d(filters=64, kernel=1) -> ReLU -> [out1]

# Branch 2: 1x1 reduce -> 3x3 conv
[b2] -> Conv2d(filters=96, kernel=1) -> ReLU
     -> Conv2d(filters=128, kernel=3, padding=1) -> ReLU -> [out2]

# Branch 3: 1x1 reduce -> 5x5 conv
[b3] -> Conv2d(filters=16, kernel=1) -> ReLU
     -> Conv2d(filters=32, kernel=5, padding=2) -> ReLU -> [out3]

# Branch 4: maxpool -> 1x1 proj
[b4] -> MaxPool(kernel=3, stride=1, padding=1)
     -> Conv2d(filters=32, kernel=1) -> ReLU -> [out4]

# Concatenate all branches along channel axis (axis=0 for C,H,W layout)
# Output: (64+128+32+32)=256 channels, 28x28
[out1, out2, out3, out4] -> Concat(axis=0) -> Output
```

Run commands:
```bash
# Visualize as SVG
mlmd visualize lenet.mlmd -o lenet.svg

# Generate PyTorch code
mlmd generate pytorch ./out

# Lint for errors
mlmd lint lenet.mlmd

# Start LSP server (for editor integration)
mlmd lsp
```

## CLI Commands

### `mlmd lint [options] <file>`
Lint a `.mlmd` file for errors (shape mismatches, cycles, missing params).
- `--json` — output diagnostics as JSON

### `mlmd visualize [options] <file>`
Generate an SVG diagram of the architecture.
- `-o, --output <file>` — output SVG file (default: stdout)

### `mlmd generate [options] <target> <outdir>`
Generate framework code from a `.mlmd` file. Reads `stdin` or detects `.mlmd` file from context.
- Targets: `pytorch`, `keras`, `candle`
- Reads `.mlmdrc` for default targets if none specified

### `mlmd lsp`
Start the LSP server for editor integration (stdin/stdout protocol).

## Configuration

Create a `.mlmdrc` JSON file in your project root:

```json
{
  "plugins": "./mlmd-plugins",
  "targets": [
    { "lang": "pytorch", "out": "./generated" },
    { "lang": "keras", "out": "./generated" }
  ]
}
```

## Syntax

- **Block declarations**: `BlockType(param=value, ...)`
- **Sequential chaining**: `Block1 -> Block2 -> Block3`
- **Tensor naming** (fork): `Block -> [tensor_name]`
- **Tensor joining**: `[tensor_a, tensor_b] -> MergeBlock`
- **Groups**: `[[ GroupPath > Component ]]`
- **Comments**: `# comment`
- **Multi-line chains**: newline after `->` continues the chain
- **Multi-line params**: params can span multiple lines inside parentheses

### Supported Block Types

Layers: Input, Output, Linear, Conv1d/2d/3d, TransposedConv2d, Embedding
Activations: ReLU, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, Sigmoid, Tanh, Softmax
Normalization: BatchNorm, LayerNorm, GroupNorm, InstanceNorm
Pooling: MaxPool, AvgPool, GlobalAvgPool, AdaptiveAvgPool
Recurrent: LSTM, GRU, RNN
Transform: Flatten, Reshape, Dropout, Pad
Merge: Add, Mul, Sub, Div, Concat, MatMul
Control flow: Split, Repeat, Map, Gather

## Examples

See the `examples/` directory:
- `lenet.mlmd` — LeNet-5
- `resnet-bottleneck.mlmd` — ResNet bottleneck with skip connection
- `attention.mlmd` — Single-head attention
- `inception.mlmd` — Inception module
- `unet.mlmd` — U-Net with encoder/decoder groups

## Editor Support

- **VS Code**: Install the `mlmd-vscode` extension from the repository
- **Zed**: Copy the `languages/mlmd/` directory to your Zed config

## Plugin System

Create custom blocks by placing `.mjs` / `.js` / `.ts` files in a plugins directory:

```typescript
// my-custom-block.mjs
export default {
  inputs: ["x"],
  outputs: ["y"],
  inferShape(inputs, params) {
    // input shapes → output shapes
    return [[...inputs[0]]];
  },
  render(ctx) { /* custom SVG rendering */ },
  codegen(ctx) { /* custom code generation */ },
};
```

Refer to the plugins directory in `.mlmdrc`:
```json
{ "plugins": "./mlmd-plugins" }
```

Shape inference, SVG rendering, and code gen will all use your plugin.

## Programmatic API

```typescript
import { tokenize, parse, buildGraph, inferShapes, getTarget } from "mlmd";
import "mlmd/blocks"; // register built-in blocks

const tokens = tokenize(source);
const { nodes } = parse(tokens);
const graph = buildGraph(nodes);
const result = inferShapes(graph, registry);
const target = getTarget("pytorch");
const files = target.generate(result.graph, registry);
```

## Development

```bash
git clone <repo>
cd mlmd
npm install
npm run build
npm test
npm run typecheck
```

## Project Status

All 33 implementation tasks complete. 298+ tests passing.
