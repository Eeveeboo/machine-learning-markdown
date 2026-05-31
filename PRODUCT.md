# MLMD (Machine Learning Markdown)

`MLMD` is a concise language for describing neural network architectures. It compiles to SVG diagrams (via `visualize`) and framework code (via `generate`).

```
mlmd visualize <file.mlmd>                     → SVG diagram
mlmd generate [target] <dir>                    → model code (target/dir optional if in .mlmdrc)
mlmd lint <file.mlmd>                           → shape validation
```

## Syntax

### Core model

Everything is a tensor dataflow graph. Blocks are nodes, named tensors are edges. Tensors are non-consuming — multiple blocks can read from the same tensor.

### Declarations

```
# Root block (no incoming edge)
BlockType (param=value, ...)

# Chain — previous block's output tensor feeds into this block
   -> BlockType (param=value, ...)

# Named tensor — creates an edge other blocks can reference
   -> [tensor_name]

# Fork — same tensor available to multiple consumers
   -> [a, b]

# Join — multiple tensors feed into one block
[a, b] -> BlockType (param=value, ...)
```

### Built-in blocks

| Category | Types |
|---|---|
| Layers | `Input`, `Output`, `Linear`, `Conv[1d|2d|3d]`, `TransposedConv2d`, `Embedding` |
| Norm | `BatchNorm`, `LayerNorm`, `GroupNorm`, `InstanceNorm` |
| Activation | `ReLU`, `SELU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU` |
| Pool | `MaxPool`, `AvgPool`, `GlobalAvgPool`, `AdaptiveAvgPool` |
| Recurrent | `LSTM`, `GRU`, `RNN` |
| Transform | `Flatten`, `Reshape`, `Dropout`, `Pad` |
| Merge | `Add`, `Concat`, `Mul`, `Sub` |

### Parameters

| Type | Examples |
|---|---|
| Number | `kernel_size=3`, `stride=2.0` |
| String | `padding="same"` |
| Bareword | `padding=same` (quotes optional) |
| Boolean | `bias=true` |
| Shape | `shape=(c,h,w)` — rank convention: 1→(w), 2→(h,w), 3→(c,h,w), 4→(b,c,h,w) |

Shapes are inferred from layer params. `mlmd lint` validates shape compatibility.

### Groups (visual containers)

```
[[ Encoder ]]
[[ ResNet50 > Stage 1 > Block A ]]     # nested
```

## Examples

### LeNet-5 (simple chain)

```
Input (shape=(1, 28, 28))
    -> Conv2d (kernel_size=5, filters=6)    -> Tanh ()
    -> AvgPool (kernel_size=2, stride=2)
    -> Conv2d (kernel_size=5, filters=16)   -> Tanh ()
    -> AvgPool (kernel_size=2, stride=2)
    -> Flatten () -> Linear (out_features=120) -> Tanh ()
    -> Linear (out_features=84) -> Tanh ()
    -> Linear (out_features=10) -> Softmax ()
```

### ResNet bottleneck (fork + join)

```
[[ ResNet50 > Bottleneck ]]

Input (shape=(256, 56, 56)) -> [main, skip]

main -> Conv2d (kernel_size=1, filters=64)  -> BatchNorm () -> ReLU ()
    -> Conv2d (kernel_size=3, filters=64, padding=same) -> BatchNorm () -> ReLU ()
    -> Conv2d (kernel_size=1, filters=256)  -> BatchNorm () -> [main_out]

skip -> Conv2d (kernel_size=1, filters=256) -> BatchNorm () -> [skip_out]

[main_out, skip_out] -> Add () -> ReLU ()
```

### Attention head (multi-input)

```
Input (shape=(512, 64)) -> [query]
Input (shape=(512, 64)) -> [key]
Input (shape=(512, 64)) -> [value]

query -> Linear (out_features=64, bias=false) -> [q_proj]
key   -> Linear (out_features=64, bias=false) -> [k_proj]
value -> Linear (out_features=64, bias=false) -> [v_proj]

[q_proj, k_proj] -> MatMul () -> Mul (scalar=0.125) -> Softmax (dim=-1) -> [attn]

[attn, v_proj] -> MatMul () -> Linear (out_features=64)
```

### Inception (1→N fork)

```
Input (shape=(192, 28, 28)) -> [x]

x -> Conv2d (1, 64)                           -> [p1]
x -> Conv2d (1, 96)  -> Conv2d (3, 128, padding=same)  -> [p2]
x -> Conv2d (1, 16)  -> Conv2d (5, 32, padding=same)  -> [p3]
x -> MaxPool (3, stride=1, padding=same) -> Conv2d (1, 32) -> [p4]

[p1, p2, p3, p4] -> Concat ()
```

### U-Net (skip connections)

```
Input (1, 572, 572) -> Conv2d (3, 64, padding=valid) -> ReLU () -> [l1_f, l1_o]

l1_o -> Conv2d (3, 64, padding=valid) -> ReLU () -> MaxPool (2, 2)
    -> Conv2d (3, 128, padding=valid) -> ReLU () -> [l2_f, l2_o]

l2_o -> Conv2d (3, 128, padding=valid) -> ReLU () -> MaxPool (2, 2)
    -> Conv2d (3, 256, padding=valid) -> ReLU () -> Conv2d (3, 256, padding=valid) -> ReLU ()
    -> ConvTransposed2d (2, 128, stride=2) -> [up1]

[l2_f, up1] -> Concat () -> Conv2d (3, 128, padding=valid) -> ReLU ()
    -> Conv2d (3, 128, padding=valid) -> ReLU () -> ConvTransposed2d (2, 64, stride=2) -> [up2]

[l1_f, up2] -> Concat () -> Conv2d (3, 64, padding=valid) -> ReLU ()
    -> Conv2d (3, 64, padding=valid) -> ReLU () -> Conv2d (1, 2)
```

## Custom blocks

Drop a `.ts` file into `blocks/<name>.ts` next to your `.mlmd`:

```ts
import { BlockPlugin } from "mlmd";

export default class Reparameterize implements BlockPlugin {
  inputs  = ["mu", "logvar"];
  outputs = ["z"];

  render(ctx: RenderContext): void { /* draw custom SVG */ }
  codegen(ctx: CodegenContext): string { /* return framework code */ }
}
```

## Config (.mlmdrc)

A `.mlmdrc` file in the project root sets defaults for CLI and LSP:

```json
{
  "plugins": "./src/mlmd-blocks",
  "targets": [
    { "lang": "pytorch",   "out": "./gen/pytorch" },
    { "lang": "candle",    "out": "./gen/rust" }
  ]
}
```

- `plugins` — path to custom block plugin directory (enables LSP autocomplete for them)
- `targets` — `generate` uses these when no CLI args given

## Editor support

### Language server

An LSP server (`mlmd lsp`) wraps the parser and provides IDE features in both VS Code and Zed:

- Completions for built-in and custom block types (from `plugins` dir)
- Diagnostics on shape mismatches, missing params, undefined tensor refs
- Hover docs on block types and params
- Go-to-definition on tensor references

### Syntax highlighting

A TextMate grammar (`syntaxes/mlmd.tmLanguage.json`) provides highlighting. Distributed as:

- **VS Code**: extension (`mlmd-vscode/`) bundling both grammar and LSP client
- **Zed**: via `languages/mlmd/` config, same grammar + LSP adapter

## Output

`mlmd visualize` produces a standalone `.svg` with group bounding boxes, tensor-annotated edges, and parameter counts on each block.
