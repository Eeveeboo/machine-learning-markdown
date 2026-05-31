# MLMD (Machine Learning Markdown)

`MLMD` is a human-redable intermediary language for describing machine learning/neural network architecture inspired by `DBML`.

`MLMD` is a concice, structured language designed to be deterministically translated into a beautiful diagram which visually describes a neural network, how the data flows through it and parameter counts.

Ideal for quickly prototyping and communicating new architectual ideas, you are encouraged to add your own never heard of before blocks into the architecture. Most common blocks (Linear, Conv, RELU, SELU) are builtin already.

There are two commands which come alongside `MLMD`

1. `mdml visualise <file.mlmd>` - this converts your mdml into a beautiful SVG and creates (TYPE UNDECIDED) files to define how your custom blocks are: a) generated into the visualisation and b) translated into pytorch, tensorflow, or candle 
2. `mdml generate <pytorch|tensorflow|candle> <directory>`

Syntax:
```
[[ Group Header ]] #many blocks can linearly be under a single group header
<BlockName> (<pram_name> = <pram_value>, ...)
    -> <AlotherBlockNameChaninedInlineWithInferedInAndOut> # Indented is treated as part of the previous line, outputs can be chained with ->
    -> [<output_vector_1>(output_tensor_size), <output_vector_2>(output_tensor_size)] # Or explicit outputs can be defined, if multiple are specified, output_tensor_size is required on each.
```
