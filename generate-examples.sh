# First build the CLI, then regenerate all example outputs
# Run from repo root after `npm install && npm run build`

npm run build
npm link

mlmd visualize examples/attention.mlmd -o examples/attention.svg
mlmd visualize examples/inception.mlmd -o examples/inception.svg
mlmd visualize examples/lenet.mlmd -o examples/lenet.svg
mlmd visualize examples/resnet-bottleneck.mlmd -o examples/resnet-bottleneck.svg
mlmd visualize examples/unet.mlmd -o examples/unet.svg

mlmd generate examples/attention.mlmd -o examples -t candle
mlmd generate examples/inception.mlmd -o examples -t candle
mlmd generate examples/lenet.mlmd -o examples -t candle
mlmd generate examples/resnet-bottleneck.mlmd -o examples -t candle
mlmd generate examples/unet.mlmd -o examples -t candle

mlmd generate examples/attention.mlmd -o examples -t pytorch
mlmd generate examples/inception.mlmd -o examples -t pytorch
mlmd generate examples/lenet.mlmd -o examples -t pytorch
mlmd generate examples/resnet-bottleneck.mlmd -o examples -t pytorch
mlmd generate examples/unet.mlmd -o examples -t pytorch
