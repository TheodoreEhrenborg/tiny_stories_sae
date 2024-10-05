# Training an Autoencoder

Let's forget about the sparse
part for now. What we want is a neural
network that takes some input and outputs
the same thing.

## What's the input?

Anthropic trained on the activation after a middle layer of Claude:

> we focused on applying SAEs to residual stream activations halfway through the model (i.e. at the “middle layer”).

They have various justifications for this. My intuition is that the middle layer is maybe where
the model is thinking the most about concepts, because at the front and back it has
to think more about processing input and preparing output, respectively.

TinyStories-33M has
4 transformer layers. So we
take the activation after the 2nd
layer.

Talk about getting the activation from the LLM

And the formula: Linear(Relu(Linear(activation)))

Show tensorboard graphs for SAE dim 100, 1000, and 10000

TODO Graph of reconstruction loss

TODO Graph of proportion of nonzero features

```admonish warning
Pytorch makes it easy to save models
using `torch.save(model, path)`.

This is a bad idea.

If you 
later move the import class to a different 
file, you may not be able to load old
checkpoints.
The [safe](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)
but clunkier way is
`torch.save(model.state_dict(), path)`.
```
