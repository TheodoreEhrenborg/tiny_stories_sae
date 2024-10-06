# Making it sparse

An autoencoder on its own is not so useful for interpreting the LLM.
Just as the LLM learned incomprehensible ways of encoding
text into 768 neural activations, the autoencoder will learn incomprehensible ways
of encoding those neural activations into its \\(F\\) features.

Here's how we encourage the autoencoder to make human-comprehensible features:
- First we make its job much easier by setting \\(F\\) to be much larger than 768.
  As we saw [earlier](training_an_autoencoder.md#training), the autoencoder can't 
  perfectly preserve activations when \\(F < 768\\), but it does ok when \\(F = 1000 \\).
  Going forwards I'll instead set \\(F = 10000 \\).
- Then we make its job harder by not letting it use very many TODO


Q: I don't see the motivation for these steps. Why is this expected to be helpful?<br>
A: It'll be less mysterious once we go through some theory on how LLMs might think.

## Theory for how LLMs might think

Under some plausible assumptions

(see Anthropic's previous work here TODO for a longer explanation)

Assumption 1: The LLM processes information in a sparse way. That is,
it knows about thousands of human-understandable topics, but
but an individual sentence only triggers a few of them at a time.

If the input text is "Mary had a little lamb", the LLM's internal representation 
is something like 
`[contains_sheep = 0.5, female_character = 1, male_character = 0, contains_dog = 0, ...]`,
where the `...` is a long list of items that _aren't_ true about the story.
Let's call these hypothesized items _concepts_, 
to distinguish them from the \\(F\\) definitely real _features_
in the sparse autoencoder's hidden layer. The end goal is to train the
autoencoder so that we can read off the concepts from the features.
These concepts 

Assumption 2: The superposition hypothesis---
the concepts are
in fact vectors in \\(\mathbb{R}^{768}\\), and
the LLM stores information as a linear sum
of near-perpendicular vectors.


When humans look at the activations, we just see
the sum of all concept vectors, and there's
no easy way for humans to read that representation.
No one neuron of the 768 neurons is the `contains_sheep` concept; it's spread across all neurons.

Note that it's possible to fit [exponentially](https://mathoverflow.net/a/24887)
[more](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) than 768 near-perpendicular vectors in 
\\(\mathbb{R}^{768}\\), so the LLM isn't limited to 768 concepts.

## Theory for how the sparse autoencoder should work

The sparse autoencoder has to transform the LLM activation \\(l \in \mathbb{R}^{768} \\) into a vector of features
where only a few features are nonzero, and then recover \\(l\\).
A plausible way to do this is:
- Use `encoder_linear` as 10000 linear probes. Each probe has two parts: \\(a_i \in \mathbb{R}^{768} \\) 
  that points in the same direction as a concept vector, and a bias \\(b_i \in \mathbb{R} \\). 
  Hence the ith autoencoder feature is \\( a_i \cdot l + b_i \\). Note that \\(b_i \\) is probably negative
  (all of this is theory), so if \\( l \\) doesn't have a large component in the \\(a_i\\) direction,
  the ReLU will set the feature to 0.
- Now only a few of the features are nonzero, satisfying the L1 penalty.
- `decoder_linear` just reverses `encoder_linear`, mapping the ith feature back to the direction
   \\(a_i \in \mathbb{R}^{768} \\).
- This reverse mapping isn't perfect, since most features were set to 0 by the ReLU.
  But those features were associated with concepts
  that the LLM wasn't thinking about (that's the sparseness assumption). So not much of \\(l\\) is lost
  from the ReLU.


The autoencoder has to reuse these features across all the examples in the training data. 
Hence the features must be synced with the concepts. So if we look at training examples
where a certain feature was strongly activated, we should see a specific pattern.
  


teasing apart the one vector in \\(\mathbb{R}^{768}\\) into the constituent items, 
seeing that only `contains_sheep` and `female_character` are true, and setting two
TODO

the LLM's internal representation



break apart the 



it




Careful when getting the magnitudes for the penalty: If the other dimension isn't summed over first, the tensor ends up being very large

seq_len
768
number of features (e.g. 10000)

Proportion of nonzero features

TODO Graph comparing the 3? 4? L1 weights

Then graph of the one that went on for a long time
