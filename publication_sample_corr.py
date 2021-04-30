import os
import dill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


def plot_correlation_nll(traces_dict, figsize=(8,5)):

    nll_mean_drift = traces_dict['nll-mean-drift']
    nll_mean_nondrift = traces_dict['nll-mean-nondrift']

    if 'nll-var-drift' in list(traces_dict.keys()):
        has_var = True
    else:
        has_var = False

    y_target = np.linspace(0.25,0.95,nll_mean_drift.shape[0])

    nll_mean_perturbed = nll_mean_drift
    nll_mean_unperturbed = nll_mean_nondrift

    if has_var:
        nll_var_perturbed = traces_dict['nll-var-drift']
        nll_var_unperturbed = traces_dict['nll-var-nondrift']

        pearson_corr_1 = pearsonr(nll_var_perturbed, nll_var_unperturbed)[0]

    pearson_corr_0 = pearsonr(nll_mean_perturbed, nll_mean_unperturbed)[0]

    # actual plotting
    nondrift_color = "tab:orange"
    drift_color = "tab:blue"

    fig, axes = plt.subplots(2,2, figsize=figsize)
    axes= axes.flatten()

    # nll_perturbed
    axes[0].plot(y_target, nll_mean_unperturbed, color=nondrift_color)
    axes_0_twin = axes[0].twinx()
    axes_1_twin = axes[1].twinx()
    axes_0_twin.plot(y_target, nll_mean_perturbed, color=drift_color)

    if has_var:
        axes[1].plot(y_target, nll_var_unperturbed,  color=nondrift_color)
        axes_1_twin.plot(y_target, nll_var_perturbed, color=drift_color)
        axes[3].scatter(nll_var_perturbed, nll_var_unperturbed, c=y_target)

    # bottom plots
    axes[2].scatter(nll_mean_perturbed, nll_mean_unperturbed, c=y_target)

    # set labels symbols
    nondrift_text = r"$S^{\bot{shift}}$"
    drift_text = r"$S^{shift}$"
    e_nll_text = r"$\mathrm{E_{\theta}(NLL)}$"
    var_nll_text = r"$\mathrm{Var_{\theta}(NLL)}$"

    # set labels
    axes[0].set_ylabel(e_nll_text+","+nondrift_text, color=nondrift_color)
    axes_0_twin.set_ylabel(e_nll_text+","+drift_text, color=drift_color)
    axes[1].set_ylabel(var_nll_text+","+nondrift_text, color=nondrift_color)
    axes_1_twin.set_ylabel(var_nll_text+","+drift_text, color=drift_color)
    axes[0].tick_params(axis='y', labelcolor=nondrift_color)
    axes_0_twin.tick_params(axis='y', labelcolor=drift_color)
    axes[1].tick_params(axis='y', labelcolor=nondrift_color)
    axes_1_twin.tick_params(axis='y', labelcolor=drift_color)

    # axes[0].set_xticks([0.25, 0.35, 0.45, 0.55,0.65,0.75,0.85,0.95])
    # axes[1].set_xticks([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    axes[0].set_xticks([0.25, 0.50, 0.75, 0.95])
    axes[1].set_xticks([0.25, 0.50, 0.75, 0.95])

    axes[0].set_xlabel("Degradation")
    axes[1].set_xlabel("Degradation")

    axes[2].set_xlabel(e_nll_text+","+drift_text)
    axes[2].set_ylabel(e_nll_text+","+nondrift_text)

    axes[3].set_xlabel(var_nll_text+","+drift_text)
    axes[3].set_ylabel(var_nll_text+","+nondrift_text)

    axes[2].text(.5, .9, "Pearson: %.2f" % pearson_corr_0,
            horizontalalignment='center',
            transform=axes[2].transAxes)

    if has_var:
        axes[3].text(.5, .9, "Pearson: %.2f" % pearson_corr_1,
                horizontalalignment='center',
                transform=axes[3].transAxes)

    # set color bar
    norm = plt.Normalize(y_target.min(), y_target.max())
    cmap = plt.get_cmap("viridis")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    axins = inset_axes(axes[3],
                       width="5%",  # width = 5% of parent_bbox width
                       height="98%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=axes[3].transAxes,
                       borderpad=0,
                       )

    cbar = fig.colorbar(sm, cax=axins, ticks=[0.25, 0.50, 0.75, 0.95])

    plt.tight_layout()
    return fig

# load dill and plot
encodes = ["390ef039e07d26d77c525c258e37f43f7a7700.p",
"96241a6129e9717884aa61338475aa228971.p",
"8ea85cad5702975aaed0a3cc1112076121d83c.p",
"a3ce2c815decfed70ad14f74666c82c1de7.p"
 ]

ml_exp_folder = "MLEXP-Explainability/"

# handle figure output folder
output_folder = "publication/"
output_prefixes = ["pronostia-bae-central-sample-",
                   "pronostia-bae-coalition-sample-",
                   "zema-bae-central-sample-",
                   "zema-bae-coalition-sample-"
                   ]
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# save figure
for encode,output_prefix in zip(encodes,output_prefixes):
    traces_dict = dill.load(open(ml_exp_folder + encode, "rb"))
    fig = plot_correlation_nll(traces_dict, figsize=(8,5))
    fig.savefig(output_folder+output_prefix+encode.split('.p')[0]+".png", dpi=500)









