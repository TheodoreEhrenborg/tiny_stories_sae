# Automatic feature examination

Manually judging if features are interpretable
doesn't scale. So I've outsourced the work
to GPT-4o, with the following prompts:

System prompt:

> You are a helpful assistant.

User prompt:

> I'll send you a sequence of texts,
> where tokens have been highlighted using Unicode Block Elements.
> Look for a pattern in which tokens get strong highlights.
> Rank how clear the pattern is on a 1-5 scale, where 1 is no detectable pattern,
> 3 is a vague pattern with some exceptions, and 5 is a clear pattern with no exceptions.
> Focus on the pattern in the strong highlights. Describe the pattern in 10 words or less"


Let's also ask GPT-4o to rank how 
interpretable the original language model activations are
(same prompt).
Hence we can compare to the sparse autoencoder's features, 
which ideally GPT-4o will rate as much more clear.

```admonish
Finicky detail: The LM activations can be positive or negative, 
unlike the always nonnegative autoencoder features. 
We want the activations to be nonnegative, so that
0 will be "weakly activating" and a large positive number
will be "strongly activating".

Three ways we could do this:
- Take the ReLU, i.e. set negative activations to 0
- Take the absolute value
- Use an affine shift, i.e. add the smallest possible number
  to all the acitvations so that the least activation is now 0
  
I don't think there's a strong justification for choosing one
of these methods over another, so I've implemented them all.
```

Results:

![GPT-4o's ranking of sparse autoencoder features and LM neurons](assets/llm_and_sae_clearness.svg)

## Discussion


The results are directionally as expected: GPT-4o ranks the 
autoencoder features as clearer: less likely to be 3/5 than the LLM activations, more likely to be 5/5.



grain of salt: gives high clearness to something that isn't clear

affine, feature 0, example 0


My anecdotal impression from looking through the results is that GPT-4o is
a lenient grader.


> Once ▄ upon ▃ a ▃ time ▄, ▄ there ▄ was ▄ a ▄ huge ▃ bear ▃ called ▄ Peter ▄. ▄ Peter ▄ lived ▃ deep ▂ in ▄ the ▄ forest ▅ in ▄ a ▄ big ▃, ▄ cos ▄y ▂ cave ▄. ▄ ▄ ▄One ▄ day ▅, ▄ Peter ▄ was ▄ feeling ▃ tired ▅ and ▄ decided ▄ he ▄ should ▃ go ▅ to ▆ sleep ▆. ▄ So ▅ he ▅ climbed ▄ into ▆ his ▅ bed ▆ and ▅ closed ▄ his ▄ eyes ▆. ▄ Suddenly ▅, ▅ he ▅ heard ▆ a ▆ noise ▆ outside █ his ▄ cave ▅. ▄ " ▅What ▅ was ▆ that ▄?" ▄ he ▅ thought ▄. ▅ ▅ ▅Peter ▄ went ▄ outside ▆ to ▅ investigate ▅. ▅ He ▆ saw ▅ a ▄ little ▄ mouse ▃, ▄ and ▄ the ▅ mouse ▄ was ▄ in ▃ trouble ▅! ▄ Its ▄ tail ▄ was ▄ stuck ▃ under ▄ a ▃ big ▄ rock ▄ and ▄ it ▄ couldn ▄'t ▅ get ▄ free ▄. ▄ ▅ ▄" ▅Don ▄'t ▃ worry ▃," ▃ said ▅ Peter ▄. ▄ " ▅I ▄'ll ▂ help ▅ you ▃!" ▄ He ▅ used ▅ all ▄ his ▅ strength ▃ to ▄ move ▅ the ▄ huge ▄ rock ▄ and ▅ the ▅ mouse ▃ ran ▄ away ▄. ▄ ▅ ▄Peter ▄ smiled ▃, ▄ but ▄ he ▅ was ▄ very ▄ tired ▄ now ▄. ▄ He ▅ said ▅ goodbye ▅ to ▆ the ▅ mouse ▂ and ▄ went ▄ back ▅ to ▅ his ▅ cave ▅. ▄ He ▅ was ▄ so ▄ exhausted ▄, ▃ he ▄ fell ▃ asleep ▅ right ▄ away ▄. ▄ ▅ ▄From ▄ that ▃ day ▄ on ▄, ▄ Peter ▄ and ▅ the ▅ mouse ▃ were ▄ good ▃ friends ▃. ▄ They ▅ always ▄ looked ▄ out ▇ for ▄ each ▃ other ▃ and ▄ helped ▄ each ▂ other ▄ if ▁ they ▅ got ▄ into ▅ trouble ▄. ▄ And ▄ they ▅ both ▄ slept ▄ well ▄ each ▃ night ▅, ▄ knowing ▄ they ▄ had ▄ a ▄ friend ▄ who ▅ would ▃ help ▄ them ▄ out ▄ if ▂ they ▅ needed ▅ it ▄! ▄

    {
      "scratch_work": "Upon reviewing the text, the strongest highlights appear consistently around dialogue sections, especially exclamations or important dialogue elements. This pattern is evident across different examples.",
      "clearness": 4,
      "short_pattern_description": "Strong highlights emphasize dialogue and reactions.",
      "feature_idx": 0
    },




## Notable neurons

## Comparing to raw LLM neurons



TODO Note that what I call neurons
are the residual stream activations. 
i.e. the exact tensors I trained the SAE on 

Anthropic
compared their autoencoder's interpretability
to neurons in Claude's previous MLP layer.


My prior is that the MLP neurons wouldn't be interpretable either.


Go through some examples in detail to show that GPT-4o is wrong

It'd be nice if we could see how the SAE gets better over training time

Really we need a less noisy metric than this. Ideas for improvements:

- Reading Unicode block elements might be tricky for the LLM
- Tell it "be strict"
- Change the question. First ask it to figure out the theme, then ask a different LLM to rank how closely the highlighting matches the theme, or ask it to use the theme to distinguish highlighting-using-that-feature from highlighting-that-doesn't-use-that-feature
- Increase sample size until clearly statistically significant
