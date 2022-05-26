import re
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from celer.plot_utils import configure_plt


SAVEFIG = False
# SAVEFIG = True
figname = "mcp"
# figname = "finance"
# figname = "rcv1_news20"

# RUN `benchopt run . --config config_medium.yml`, then replace BENCH_NAME
# by the name of the produced results csv file.

BENCH_NAME = "mcp-neurips.csv"

FLOATING_PRECISION = 1e-8
MIN_XLIM = 1e-3
MARKERS = list(plt.Line2D.markers.keys())[:-4]


SOLVERS = {
    'cd': 'coordinate descent',
    'GIST': 'GIST',
    'PGD[use_acceleration=False]': 'PGD',
    'PGD[use_acceleration=True]': 'PGD accelerated',
    'WorkSet_CD': 'WorkSetCD',
    'skglm': 'skglm',
}

all_solvers = SOLVERS.keys()

DICT_XLIM = {
    "libsvm[dataset=rcv1.binary]": 1e-2,
    "libsvm[dataset=news20.binary]": 1e-1,
    "libsvm[dataset=colon-cancer]": 1e-4,
    "libsvm[dataset=gisette]": 1e-2,
}

DICT_TITLE = {
    'MCP Regression[gamma=3,reg=0.1]': r'$\lambda = 0.1 \lambda_{\mathrm{max}}$',
    'MCP Regression[gamma=3,reg=0.01]': r'$\lambda = 0.01 \lambda_{\mathrm{max}}$',
    'MCP Regression[gamma=3,reg=0.001]': r'$\lambda = 0.001 \lambda_{\mathrm{max}}$',
}

DICT_YLABEL = {
    'libsvm[dataset=colon-cancer]': "colon-cancer",
    'Simulated[n_features=2000,n_samples=500,scale=True]': 'simulated',
    'Simulated[n_features=2000,n_samples=500,scale=False]': 'simulated (scaled)'
}

DICT_YTICKS = {
    'libsvm[dataset=colon-cancer]': [1e3, 1, 1e-3, 1e-6],
    'Simulated[n_features=2000,n_samples=500,scale=True]': [1e3, 1, 1e-3, 1e-6],
    'Simulated[n_features=2000,n_samples=500,scale=False]': [1e3, 1, 1e-3, 1e-6],
}

DICT_XTICKS = {
    'Simulated[n_features=2000,n_samples=500,scale=False]': np.geomspace(1e-2, 1e2, 5),
    'Simulated[n_features=2000,n_samples=500,scale=True]': np.geomspace(1e-2, 1e2, 5),
    'libsvm[dataset=colon-cancer]': np.geomspace(1e-4, 1e2, 7),
}

configure_plt()
CMAP = plt.get_cmap('tab20')
style = {solv: (CMAP(i), MARKERS[i]) for i, solv in enumerate(all_solvers)}


df = pd.read_csv(BENCH_NAME, header=0, index_col=0)


solvers = df["solver_name"].unique()
solvers = np.array(sorted(solvers, key=lambda key: SOLVERS[key].lower()))
datasets = [
    'libsvm[dataset=colon-cancer]',
    'Simulated[n_features=2000,n_samples=500,scale=False]',
    'Simulated[n_features=2000,n_samples=500,scale=True]',
]

objectives = df["objective_name"].unique()

titlesize = 22
ticksize = 16
labelsize = 20
regex = re.compile('\[(.*?)\]')

plt.close('all')
fig1, axarr = plt.subplots(
    len(datasets),
    len(objectives),
    sharex=False,
    sharey='row',
    figsize=[11, 1 + 2 * len(datasets)],
    constrained_layout=True)

# handle if there is only 1 dataset/objective:
if len(datasets) == 1:
    if len(objectives) == 1:
        axarr = np.atleast_2d(axarr)
    else:
        axarr = axarr[None, :]
elif len(objectives) == 1:
    axarr = axarr[:, None]

for idx_data, dataset in enumerate(datasets):
    df1 = df[df['data_name'] == dataset]
    for idx_obj, objective in enumerate(objectives):
        df2 = df1[df1['objective_name'] == objective]
        ax = axarr[idx_data, idx_obj]

        c_star = np.min(df2["objective_value"]) - FLOATING_PRECISION
        for i, solver_name in enumerate(solvers):
            df3 = df2[df2['solver_name'] == solver_name]
            curve = df3.groupby('stop_val').median()
            y = curve["objective_opt_violation"]

            ax.loglog(
                curve["time"], y, color=style[solver_name][0],
                marker=style[solver_name][1], markersize=6,
                label=SOLVERS[solver_name], linewidth=2, markevery=3)

        ax.set_xlim([DICT_XLIM.get(dataset, MIN_XLIM), ax.get_xlim()[1]])
        axarr[len(datasets)-1, idx_obj].set_xlabel("Time (s)", fontsize=labelsize)
        axarr[0, idx_obj].set_title(
            DICT_TITLE[objective], fontsize=labelsize)

        ax.grid()
        ax.set_xticks(DICT_XTICKS[dataset])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)

    if regex.search(dataset) is not None:
        dataset_label = (regex.sub("", dataset) + '\n' +
                         '\n'.join(regex.search(dataset).group(1).split(',')))
    else:
        dataset_label = dataset
    axarr[idx_data, 0].set_ylabel(DICT_YLABEL[dataset], fontsize=labelsize)
    axarr[idx_data, 0].set_yticks(DICT_YTICKS[dataset])

plt.show(block=False)


fig2, ax2 = plt.subplots(1, 1, figsize=(20, 4))
n_col = 3
if n_col is None:
    n_col = len(axarr[0, 0].lines)

# take first ax, more likely to have all solvers converging
ax = axarr[0, 0]
lines_ordered = list(itertools.chain(
    *[ax.lines[i::n_col] for i in range(n_col)]))
legend = ax2.legend(
    lines_ordered, [line.get_label() for line in lines_ordered], ncol=n_col,
    loc="upper center")
fig2.canvas.draw()
fig2.tight_layout()
width = legend.get_window_extent().width
height = legend.get_window_extent().height
fig2.set_size_inches((width / 80,  max(height / 80, 0.5)))
plt.axis('off')
plt.show(block=False)


if SAVEFIG:
    fig1_name = f"figures/{figname}.pdf"
    fig1.savefig(fig1_name)
    os.system(f"pdfcrop {fig1_name} {fig1_name}")
    fig1.savefig(f"figures/{figname}.svg")

    fig2_name = f"figures/{figname}_legend.pdf"
    fig2.savefig(fig2_name)
    os.system(f"pdfcrop {fig2_name} {fig2_name}")
    fig2.savefig(f"figures/{figname}_legend.svg")
