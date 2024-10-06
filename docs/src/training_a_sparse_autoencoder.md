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
and compensating by making the elements \\(C_{ij}\\) of `decoder_linear`'s weight matrix large.

So instead we broadcast multiply \\(f\\) by \\(C\\) to get a list of 10000 feature vectors,
each one in \\( \mathbb{R}^{768}  \\). Then we collapse each of those feature vectors into a single magnitude using
the L2 norm,
and then apply the L1 penalty to that list of 10000 numbers.

But look at those dimensions again.
We have created a tensor with dimensions `(768, 10000)`. At 4 bytes per single-precision float, 
that's ~30 megabytes. We need one such tensor per token in the training example,
so we need 6 gigabytes when the example is 200 tokens long. All of this is batch size 1.

Luckily there's a trick: First we apply the L2 TODO 




so for a single training example, we create a mat
(easily over 200 tokens per example, which is 6 GB).
(this is batch size 1)




Careful when getting the magnitudes for the penalty: If the other dimension isn't summed over first, the tensor ends up being very large

seq_len
768
number of features (e.g. 10000)
```

Proportion of nonzero features

TODO Graph comparing the 3? 4? L1 weights

Then graph of the one that went on for a long time
