# Training a sparse autoencoder

## Tuning the L1 penalty

Let's get back to experiments. We want to train an autoencoder while minimizing
the sum of the L2 reconstruction loss and the L1 sparsity penalty. The theory 
doesn't tell us what their relative importance should be. So the loss function is really
\\[
\mathrm{loss} = \mathrm{L2\\_reconstruction\\_loss} + \lambda \cdot \mathrm{L1\\_sparsity\\_penalty}
\\]

where \\( \lambda \\) is an unknown hyperparameter. If \\( \lambda \\) is too small, we've just trained a
non-sparse autoencoder. If it's too large, gradient descent will prevent any feature from
ever being nonzero.


sweep




```admonish
Which tensor do we apply the L1 penalty too? We can't just apply it to the feature vector
\\(f \in \mathbb{R}^{10000} \\).
In that case the autoencoder could cheat the L1 penalty by making all the features \\(f_i\\) small but nonzero,
and compensating by making the elements \\(C_{ij}\\) of `decoder_linear` large.
So instead we broadcast

But now this




Careful when getting the magnitudes for the penalty: If the other dimension isn't summed over first, the tensor ends up being very large

seq_len
768
number of features (e.g. 10000)
```

Proportion of nonzero features

TODO Graph comparing the 3? 4? L1 weights

Then graph of the one that went on for a long time
