import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas
import seaborn as sns
from beartype import beartype


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--response_json", type=Path, required=True)
    parser.add_argument("--output_png", type=str, default="/results/out.png")
    return parser


def main(args: Namespace):
    sns.set_theme()

    data = json.load(args.response_json.open())
    df = pandas.DataFrame(data["responses"])
    seaborn_plot = sns.countplot(df, x="clearness")
    seaborn_plot.set(
        xlabel="GPT-4's ranking of clearness",
        title="LLM activations, put through absolute value",
    )
    fig = seaborn_plot.get_figure()
    fig.savefig(args.output_png)


if __name__ == "__main__":
    main(make_parser().parse_args())
