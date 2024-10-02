# Training an Autoencoder

Talk about getting the activation from the LLM
  
And the formula: Linear(Relu(Linear(activation)))
  

Show tensorboard graphs for SAE dim 100, 1000, and 10000

```admonish warning
Pytorch makes it easy to save models
using `torch.save(model, path)`.

This is a bad idea: If you 
later move the import class to a different 
file, you may not be able to load old
checkpoints.
The [safe](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)
but clunkier way is
`torch.save(model.state_dict(), path)`.
```
