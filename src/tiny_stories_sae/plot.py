import json

import pandas
import seaborn as sns

sns.set_theme()

data = json.load(open("/results/gpt4_api/20240930-195450llm_abs_full"))
df = pandas.DataFrame(data["responses"])
seaborn_plot = sns.countplot(df, x="clearness")
seaborn_plot.set(xlabel="GPT-4's ranking of clearness")
fig = seaborn_plot.get_figure()
fig.savefig("/results/out.png")
