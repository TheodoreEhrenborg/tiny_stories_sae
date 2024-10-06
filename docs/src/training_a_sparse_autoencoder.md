# Training a sparse autoencoder

## Tuning the L1 penalty

Let's get back to experiments. We want to train an autoencoder while minimizing
the sum of the L2 reconstruction loss and the L1 sparsity penalty. The theory above
doesn't tell us what their relative importance should be. So the loss function is really

where `lambda` is an unknown hyperparameter. If `lambda` is too small, we've just trained a
non-sparse autoencoder. If it's too large, gradient descent will prevent any feature from
ever being nonzero.



\\( \lambda \\)



```admonish
Careful when getting the magnitudes for the penalty: If the other dimension isn't summed over first, the tensor ends up being very large

seq_len
768
number of features (e.g. 10000)
```

Proportion of nonzero features

TODO Graph comparing the 3? 4? L1 weights

Then graph of the one that went on for a long time
