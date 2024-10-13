# Manual feature examination

Look at features 0, 1, 2

all the way to 10?


On 1000 examples from the validation set,
each of the autoencoders 10000 features activated at least once
i.e. there were no dead features.

My understanding of the
[literature](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#scaling-sae-experiments) 
is that dead features
are more of a problem for larger autoencoders:

> At the end of training, we defined “dead” features as those which were not active over a sample of 10^{7} tokens. The proportion of dead features was roughly 2% for the 1M SAE, 35% for the 4M SAE, and 65% for the 34M SAE.




I'm going to drag you through 10 example
features. Why so many?  TODO Fix

(We can outsource this work to ChatGPT, but
for this project I think ChatGPT did a much 
worse job of ranking how specific the features) TODO Fix


The goal here is monosemanticity: 
We'd like each feature of the autoencoder to 
activate on input text with _one_ narrow theme, like
"cake/pie" or "numerical digits". 

I'm only looking at the top 10 examples (ranked by how
strongly the feature activates) for each feature. TODO Fix


(I'll look in more depth at feature 6 later) TODO Fix


### Feature 0

<details>
<summary>SUMMARY</summary>

DETAILS

</details>

### Feature 1

### Feature 2

## Comparing to raw LLM neurons

My prior is that the LLM activations
won't correspond to specific topics.

But we should give them the benefit of the doubt.

There are three ways we could interpret the activations:

Go through some examples in detail to show that ChatGPT is wrong

### Feature 0
### Feature 1

### Feature 2

### Feature 3

### Feature 4

### Feature 5
### Feature 6
### Feature 7
### Feature 8
### Feature 9
