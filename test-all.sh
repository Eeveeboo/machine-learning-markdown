npm install -g mlmd

npx mlmd visualize examples/attention.mlmd -o attention.svg
npx mlmd visualize examples/inception.mlmd -o inception.svg
npx mlmd visualize examples/lenet.mlmd -o lenet.svg
npx mlmd visualize examples/resnet-bottleneck.mlmd -o resnet-bottleneck.svg
npx mlmd visualize examples/unet.mlmd -o unet.svg

npx mlmd generate examples/attention.mlmd -o .test -t candle
npx mlmd generate examples/inception.mlmd -o .test -t candle
npx mlmd generate examples/lenet.mlmd -o .test -t candle
npx mlmd generate examples/resnet-bottleneck.mlmd -o .test -t candle
npx mlmd generate examples/unet.mlmd -o .test -t candle

npx mlmd generate examples/attention.mlmd -o .test -t pytorch
npx mlmd generate examples/inception.mlmd -o .test -t pytorch
npx mlmd generate examples/lenet.mlmd -o .test -t pytorch
npx mlmd generate examples/resnet-bottleneck.mlmd -o .test -t pytorch
npx mlmd generate examples/unet.mlmd -o .test -t pytorch
