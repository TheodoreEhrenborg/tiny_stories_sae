#!/usr/bin/env python3
from jaxtyping import Float, jaxtyped, Int

from argparse import ArgumentParser, Namespace
from tiny_stories_sae.lib import get_rotation_between

import torch
from beartype import beartype
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from tiny_stories_sae.gather_high_activations import (
    format_token,
    normalize_activations,
)
from tiny_stories_sae.lib import make_base_parser, setup, get_llm_activation_from_tensor

# TODO Refactor argparse:
# Some scripts have arguments they can't use


@beartype
def main(user_args: Namespace):
    llm = AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-33M", local_files_only=user_args.no_internet
    )
    if user_args.fast:
        llm.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neo-125M", local_files_only=user_args.no_internet
    )
    tokenizer.pad_token = tokenizer.eos_token
    sample = "There once was a cat"
    input_tokens = torch.tensor(tokenizer(sample)["input_ids"]).unsqueeze(0)
    if user_args.fast:
        input_tokens = input_tokens.cuda()
    if user_args.print_unsteered:
        output_text = llm.generate(
            input_tokens,
            max_length=1000,
            num_beams=1,
            generation_config=GenerationConfig(do_sample=True, temperature=1.0),
        )
        print("Unsteered output:")
        print(tokenizer.decode(output_text[0]))

    _, steered_llm, sae, tokenizer = setup(
        user_args.sae_hidden_dim, user_args.fast, user_args.no_internet
    )
    sae = torch.load(user_args.checkpoint, weights_only=False, map_location="cpu")
    with torch.no_grad():
        encoder_vector = sae.encoder.weight[user_args.which_feature, :]
        decoder_vector = sae.decoder.weight[:, user_args.which_feature]
        print("Rotation", get_rotation_between(encoder_vector, decoder_vector))
    if user_args.fast:
        sae.cuda()
    sae.eval()
    onehot = torch.zeros(sae.sae_hidden_dim)
    onehot[user_args.which_feature] = 1
    if user_args.fast:
        onehot = onehot.cuda()
        decoder_vector = decoder_vector.cuda()
    nudge = sae.decoder(onehot)
    assert nudge.shape == torch.Size([768]), nudge.shape
    print(
        "Rotation between nudge and decoder_vec",
        get_rotation_between(nudge, decoder_vector),
    )
    assert torch.allclose(nudge, decoder_vector), (
        nudge[:10],
        decoder_vector[:10],
        (nudge - decoder_vector)[:10],
    )
    norm_nudge = decoder_vector / torch.linalg.vector_norm(decoder_vector)

    @jaxtyped(typechecker=beartype)
    def get_activation_strength(
        llm_activation: Float[torch.Tensor, "1 seq_len 768"],
    ) -> Float[torch.Tensor, "seq_len"]:
        norm_act = normalize_activations(llm_activation)
        strength = sae(norm_act)[1][0, :, user_args.which_feature]
        return strength

    def simple_nudge_hook(module, args, output):
        activation = output[0]
        print(
            "This feature's activation pre nudge", get_activation_strength(activation)
        )
        activation_with_nudge = activation + user_args.feature_strength * norm_nudge
        print(
            "This feature's activation post nudge",
            get_activation_strength(activation_with_nudge),
        )
        return activation_with_nudge, output[1]

    def nudge_hook(module, args, output):
        activation = output[0]
        print(
            "This feature's activation pre nudge", get_activation_strength(activation)
        )
        norm_act2 = normalize_activations(activation)
        norm_activation_no_nudge = norm_act2 - norm_nudge * torch.einsum(
            "i,jki->jk", norm_nudge, norm_act2
        ).unsqueeze(2)
        print(
            "This feature's activation with nudge zeroed out",
            get_activation_strength(norm_activation_no_nudge),
        )
        norm_activation_with_nudge = (
            norm_activation_no_nudge + user_args.feature_strength * norm_nudge
        )
        print(
            "This feature's activation post nudge",
            get_activation_strength(norm_activation_with_nudge),
        )
        return (
            (
                norm_activation_with_nudge
                * activation.std()
                / torch.sqrt(torch.tensor(768))
            )
            + activation.mean()
        ), output[1]

    steered_llm.transformer.h[1].register_forward_hook(simple_nudge_hook)

    steered_output_text = steered_llm.generate(
        input_tokens,
        max_length=1000,
        num_beams=1,
        generation_config=GenerationConfig(do_sample=True, temperature=1.0),
    )
    print("Steered output:")
    print(tokenizer.decode(steered_output_text[0]))

    activation = get_llm_activation_from_tensor(llm, steered_output_text)
    norm_act = normalize_activations(activation)
    _, feat_magnitudes = sae(norm_act)
    strengths = feat_magnitudes[0, :, user_args.which_feature].tolist()
    max_strength = max(strengths)
    print(
        "".join(
            format_token(tokenizer, int(token), strength, max_strength)
            for token, strength in zip(steered_output_text[0], strengths, strict=True)
        )
    )


@beartype
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--which_feature", type=int, required=True)
    parser.add_argument("--feature_strength", type=float, default=10.0)
    parser.add_argument("--print_unsteered", action="store_true")
    parser.add_argument("--no_internet", action="store_true")
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
