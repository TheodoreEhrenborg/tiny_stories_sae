# Making it sparse

An autoencoder on its own is not so useful for interpreting what the LLM
is thinking about. Just as the LLM learned incomprehensible ways of encoding
text into 768 neural activations, the autoencoder will learn incomprehensible ways
of encoding those neural activations into its \\(F\\) features.

Here's how we encourage the autoencoder to make human-comprehensible features:
- First we make its job much easier by setting \\(F\\) to be much larger than 768.
  As we saw [earlier](training_an_autoencoder.md#training), the autoencoder can't 
  perfectly preserve activations when \\(F < 768\\), but it does ok when \\(F = 1000 \\).
  Going forwards I'll instead set \\(F = 10000 \\).
- Then we make its job harder by not letting it use very many TODO


The hope is that the LLM is only thinking about a few things at a time.
If the input text is "Mary had a little lamb", the LLM's internal representation 
is something like 
`[contains_sheep = 1, female_character = 1, male_character = 0, contains_dog = 0, ...]`,
where the `...` is a long list of items that _aren't_ true about the story.
(Let's call these hypothetical items ur-features, 
to distinguish them from the \\(F)\\) definitely real features 
of the sparse autoencoder's hidden layer.)

There's no easy way for humans to read that representation, since the ur-features are
in fact vectors in \\(\mathbb{R}^{768}\\), and all we see is the sum of all ur-features.
This is the superposition hypothesis---the LLM stores information as a linear sum
of near-perpendicular vectors.
Moreover, no one neuron of the 768 neurons is the `contains_sheep` ur-feature; it's spread across all neurons.

But the sparse autoencoder has to transform the LLM activation \\(l \in \mathbb{R}^{768} \\) into a vector of features
where only a few features are nonzero. 
It can best do this by
- using `encoder_linear` as 10000 linear probes. Each probe is \\(a_i \in \mathbb{R}^{768} \\) 
  pointing in the same direction as an ur-feature, and also a bias \\(b_i \in \mathbb{R} \\).  
  Hence the ith feature is \\( a_i \cdot l + b_i \\). Note that \\(b_i \\) is probably negative
  (all of this is theory), so if \\( l \\) doesn't have a large component in the \\(a_i\\) direction,
  the ReLU will set the feature to 0.
- Now only a few of the features are nonzero, 


teasing apart the one vector in \\(\mathbb{R}^768\\) into the constituent items, 
seeing that only `contains_sheep` and `female_character` are true, and setting two
TODO

the LLM's internal representation



break apart the 



it

(see Anthropic's previous work here TODO for a longer explanation)



Careful when getting the magnitudes for the penalty: If the other dimension isn't summed over first, the tensor ends up being very large

seq_len
768
number of features (e.g. 10000)

Proportion of nonzero features

TODO Graph comparing the 3? 4? L1 weights

Then graph of the one that went on for a long time
