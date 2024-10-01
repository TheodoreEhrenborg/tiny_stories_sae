import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# sns.set_theme(style="ticks")
sns.set_theme()


# f, ax = plt.subplots(figsize=(7, 5))
# sns.despine(f)
titanic = sns.load_dataset("titanic")
print(titanic)
seaborn_plot = sns.barplot([2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5])
# ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
# ax.set_xticks([1, 2, 3, 4, 5])
fig = seaborn_plot.get_figure()
fig.savefig("/results/out.png")
