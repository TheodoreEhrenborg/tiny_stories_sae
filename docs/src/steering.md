# Steering

```admonish warning
It's tempting to instead define
`nudge = sae.decoder(onehot)`,
where `onehot = torch.tensor([0,...,0,1,0,...,0])` is a unit vector pointing in the direction
of the feature we want to amplify.

The problem is that `sae.decoder` multiplies by the weight matrix
(picking out the correct feature vector) but _also_ adds the bias.
This bias is large enough to mess up steering, i.e.
adding in this corrupted `nudge` doesn't make the model generate
text that resembles the desired feature.
```

Feature 6


The fire neuron really seems to be a danger neuron,
based on the steering results

Look at long tail to confirm this
