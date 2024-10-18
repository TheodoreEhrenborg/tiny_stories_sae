# Automatic feature examination

![SAE graph](assets/sae.png)

## Notable neurons

## Comparing to raw LLM neurons

![LLM graph using absolute value](assets/llm_abs.png)
![LLM graph using ReLU](assets/llm_relu.png)
![LLM graph using affine shift](assets/llm_affine.png)

My prior is that the LLM activations
won't correspond to specific topics.

There are three ways we could interpret the activations:

Go through some examples in detail to show that ChatGPT is wrong

It'd be nice if we could see how the SAE gets better over training time

Really we need a less noisy metric than this. Ideas for improvements:

- Reading Unicode block elements might be tricky for the LLM
- Tell it "be strict"
- Change the question. First ask it to figure out the theme, then ask a different LLM to rank how closely the highlighting matches the theme, or ask it to use the theme to distinguish highlighting-using-that-feature from highlighting-that-doesn't-use-that-feature
- Increase sample size until clearly statistically significant
