# Future works

### Channel pruning

First of all I think student network could be more compressed. I would somehow choose the architecture depending on teacher model. For example I could use structured pruning to reduce number of channels in residual blocks. I would remain downsampling and upsampling layers the same as it play an encoder-decoder role to/from feature space. The loss between hidden states, therefore, would counts through one conv layer with kernel size 1 to match number of channels.

### Teacher improvements

If I had enough time I would train my network for 200 epochs instead of 100 as it was proposed in original CycleGAN paper. But it would take about 5 days to do so.

