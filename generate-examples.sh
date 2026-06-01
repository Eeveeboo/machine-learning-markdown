# First build the CLI, then regenerate all example outputs
# Run from repo root after `npm install && npm run build`

npx mlmd visualize examples/attention.mlmd -o examples/attention.svg
npx mlmd visualize examples/inception.mlmd -o examples/inception.svg
npx mlmd visualize examples/lenet.mlmd -o examples/lenet.svg
npx mlmd visualize examples/resnet-bottleneck.mlmd -o examples/resnet-bottleneck.svg
npx mlmd visualize examples/unet.mlmd -o examples/unet.svg

npx mlmd generate examples/attention.mlmd -o examples -t candle
npx mlmd generate examples/inception.mlmd -o examples -t candle
npx mlmd generate examples/lenet.mlmd -o examples -t candle
npx mlmd generate examples/resnet-bottleneck.mlmd -o examples -t candle
npx mlmd generate examples/unet.mlmd -o examples -t candle

npx mlmd generate examples/attention.mlmd -o examples -t pytorch
npx mlmd generate examples/inception.mlmd -o examples -t pytorch
npx mlmd generate examples/lenet.mlmd -o examples -t pytorch
npx mlmd generate examples/resnet-bottleneck.mlmd -o examples -t pytorch
npx mlmd generate examples/unet.mlmd -o examples -t pytorch
