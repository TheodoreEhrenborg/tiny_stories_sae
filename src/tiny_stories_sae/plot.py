import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas

# sns.set_theme(style="ticks")
sns.set_theme()

import json

data = json.load(open("/results/gpt4_api/20240930-195450llm_abs_full"))
df = pandas.DataFrame(data["responses"])
print(df)
# f, ax = plt.subplots(figsize=(7, 5))
# sns.despine(f)
titanic = sns.load_dataset("titanic")
# print(titanic)
# seaborn_plot = sns.barplot([2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5])
seaborn_plot = sns.countplot(df, x="clearness")
# ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
# ax.set_xticks([1, 2, 3, 4, 5])
fig = seaborn_plot.get_figure()
fig.savefig("/results/out.png")
