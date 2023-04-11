import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

dataset = 'lineitem'
df = pd.read_csv(f'/home/clb/AQP/output/flow-tiny-{dataset}-spline/eval.csv')
df['selectivity'] *= 100
methods = ['flow', 'verdict', 'vae', 'deepdb']
funcs = ['cnt', 'avg', 'sum', 'var', 'std']
methods_alias = {a: b for a, b in zip(methods, ['NFAQP', 'VerdictDB', 'VAE', 'DeepDB'])}
funcs_alias = {a: b for a, b in zip(funcs, ['COUNT', 'AVG', 'SUM', 'VAR', 'STD'])}
plot_df = pd.DataFrame([], columns=['Methods', 'Selectivity', "Aggregation Functions", 'sMAPE'])
upper = [100, 10, 1, 0.1]
lower = [10, 1, 0.1, 0.01]
for l, u in zip(lower, upper):
    print(l, u)
    idx = df['selectivity'].between(l, u)
    if idx.sum() <= 0:
        continue
    sel_df = df[idx].mean()

    for m in methods:
        for f in funcs:
            if f != 'sum':
                continue;
            if m == 'deepdb' and f in ['var', 'std']:
                continue
            err_col = sel_df[f'{m}_{f}_err']
            sub_line = [[funcs_alias[f]] * 1, [u] * 1, [methods_alias[m]] * 1, [err_col] * 1]
            sub_df = pd.DataFrame(sub_line).transpose()
            sub_df.columns = plot_df.columns
            print(sub_df)
            plot_df = pd.concat([plot_df, sub_df])
        

sns.lineplot(
    data=plot_df,
    x="Selectivity", y="sMAPE", hue="Methods", style="event",
    markers=True, dashes=False, ci=False
)

plt.savefig(f"{dataset}_sel.pdf", format="pdf", dpi=1200, bbox_inches="tight")
