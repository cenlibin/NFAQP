import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

PLOT_PATH = "/home/clb/AQP/plot"

def plot_err(dataset, df):
    methods = ['flow', 'verdict', 'vae', 'deepdb']
    funcs = ['cnt', 'avg', 'sum', 'var', 'std']
    methods_alias = {a: b for a, b in zip(methods, ['NFAQP', 'VerdictDB', 'VAE', 'DeepDB'])}
    funcs_alias = {a: b for a, b in zip(funcs, ['COUNT', 'AVG', 'SUM', 'VAR', 'STD'])}

    plot_df = pd.DataFrame([], columns=['Methods', 'Aggregation Functions', 'sMAPE'])
    N = df.shape[0]
    for m in methods:
        for f in funcs:
            if m == 'deepdb' and f in ['var', 'std']:
                continue
            err_col = df [f'{m}_{f}_err']
            sub_df = pd.DataFrame([[funcs_alias[f]] * N, [methods_alias[m]] * N, err_col]).transpose()
            sub_df.columns = plot_df.columns
            plot_df = pd.concat([plot_df, sub_df])

    width = 12
    ratio = 0.2
    fig = plt.figure(figsize=(width, int(width * ratio)))
    sns.set_style("ticks")
    sns.set(font="Times New Roman")
    ax = sns.boxplot(x="Methods", y="sMAPE",
                    hue="Aggregation Functions", showfliers=False, palette="Paired",
                    data=plot_df)
    ax.set(xlabel=None)
    sns.move_legend(ax, 'lower center', bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)
    # sns.despine(offset=10, trim=True)

    plt.savefig(os.path.join(PLOT_PATH, f"./err/{dataset}_err.pdf"), format="pdf", dpi=1200, bbox_inches="tight")