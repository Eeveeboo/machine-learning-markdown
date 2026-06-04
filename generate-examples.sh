#!/usr/bin/env bash
# Regenerate all example outputs from .mlmd source files.
# Run from repo root. Requires the mlmd CLI to be built.
#
#   cargo build --release -p mlmd
#   ./generate-examples.sh
#
# Or via make:
#   make generate-examples

set -euo pipefail

MLMD="${MLMD:-cargo run -p mlmd --}"

$MLMD visualize examples/attention.mlmd -o examples/attention.svg
$MLMD visualize examples/inception.mlmd -o examples/inception.svg
$MLMD visualize examples/lenet.mlmd -o examples/lenet.svg
$MLMD visualize examples/resnet-bottleneck.mlmd -o examples/resnet-bottleneck.svg
$MLMD visualize examples/unet.mlmd -o examples/unet.svg

$MLMD generate examples/attention.mlmd -o examples -t candle
$MLMD generate examples/inception.mlmd -o examples -t candle
$MLMD generate examples/lenet.mlmd -o examples -t candle
$MLMD generate examples/resnet-bottleneck.mlmd -o examples -t candle
$MLMD generate examples/unet.mlmd -o examples -t candle

$MLMD generate examples/attention.mlmd -o examples -t pytorch
$MLMD generate examples/inception.mlmd -o examples -t pytorch
$MLMD generate examples/lenet.mlmd -o examples -t pytorch
$MLMD generate examples/resnet-bottleneck.mlmd -o examples -t pytorch
$MLMD generate examples/unet.mlmd -o examples -t pytorch
