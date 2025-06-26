import anndata as ad
import squidpy as sq
import cellcharter as cc #ML
import pandas as pd
import scanpy as sc
import scvi #ML
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything #ML
from pathlib import Path
from skimage.io import imread #ML

import torch #ML
torch.set_float32_matmul_precision("high")

# Dealing with annoying warnings
from anndata._core.aligned_df import ImplicitModificationWarning
import warnings

seed_everything(1337)
scvi.settings.seed = 1337





# Info
whatFer = "one" # "one" or  "two"

#Load data
base_folder = Path(f"/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/CellCharterAnalysis/{whatFer}fer/")
adata_file = base_folder / f"{whatFer}fer.h5ad"
out_figs = base_folder / "figures"

ST_sample = sc.read_h5ad(adata_file)








# Count cells per (mouse, cluster)

# What we plotting? mouse, condition, or tumor number
ctgory = "condition"
stack_bar_out = out_figs / f"cluster"

df_counts = (
    ST_sample.obs
    .groupby([ctgory, "cluster_cellcharter"], observed = True)
    .size()
    .reset_index(name="n_cells")
)

# Pivot so that rows = mouse, columns = cluster, values = n_cells
df_pivot = df_counts.pivot(index=ctgory,
                            columns="cluster_cellcharter",
                            values="n_cells").fillna(0)

# Convert counts → fractions (sum over clusters per mouse)
df_frac = df_pivot.divide(df_pivot.sum(axis=1), axis=0)

# Creating plot
fig, ax = plt.subplots(figsize=(8, 5))
df_frac.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")

# Customizing labels and legend
ax.set_ylabel("Fraction of cells")
ax.set_xlabel(ctgory)
ax.set_title(f"Cluster composition per {ctgory}")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")

# Layout and saving
prop_fig_out = out_figs / f"cluster_proportion_{ctgory}.png"

plt.tight_layout()
plt.savefig(prop_fig_out, dpi=300)
plt.show()



print(ST_sample.X.dtype)
print(ST_sample.layers.keys())





ST_sample.obsp


cc.gr.nhood_enrichment(
    ST_sample,
    cluster_key="cluster_cellcharter",
    connectivity_key="spatial", #Why does it add _connectivities?!?! weird.
    symmetric=False,         # asymmetric by default (enrichment i→j ≠ j→i)
    only_inter=True,         # count only links between different clusters
    pvalues=True,           # no permutations/p-values (fast, analytical)
    observed_expected=False, # only keep final enrichment, not the raw observed/expected
    copy=False               # write results into ST_sample.uns under "cluster_cellcharter_nhood_enrichment"
    seed = 1337,
    # n_jobs (Optional[int]) – Number of parallel jobs. # TODO can I use this?
)


ST_sample.obs['condition'].unique()





enrich_plot = out_figs / "Neighborhood_enrichment.png"

cc.pl.nhood_enrichment(
    ST_sample,
    cluster_key="cluster_cellcharter",
    title="Neighborhood Enrichment, All Samples",
    palette="tab10",
    save = enrich_plot
    # significance=0.005
)

# Scores typically range from −0.2 to +0.7
# Values near ±0.05 → no strong enrichment or depletion.
# Values > 0.2 or < −0.1 → likely biologically meaningful.



# ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# silence NumPy divide‐by‐zero / invalid warnings - Source is probably from no edges going to certain clusters
old_err = np.seterr(divide="ignore", invalid="ignore")

# Run the enrichment
cc.gr.diff_nhood_enrichment(
    ST_sample,
    cluster_key="cluster_cellcharter",
    condition_key="condition",
    condition_groups=["NoTx", "RTCyT72", "RTCyPSCA", "CyPSCA", "CyT72"],
    pvalues=True,
    copy=False,
)

# Restore NumPy’s old error settings
np.seterr(**old_err)

# Re‐enable warnings if you want
warnings.resetwarnings()


# ignoring "Transforming to str index."
warnings.filterwarnings(
    "ignore",
    message="Transforming to str index.",
    category=ImplicitModificationWarning
)

diff_enrichment = out_figs / "diff_enrichment_plot.png"
cc.pl.diff_nhood_enrichment(
    ST_sample,
    cluster_key="cluster_cellcharter",
    condition_key="condition",
    condition_groups=["RTCyT72", "RTCyPSCA", "CyPSCA", "CyT72", "NoTx"],
    # title=" vs NoTx",
    ncols=3,
    palette="tab10",
    save = diff_enrichment
)

# returning functionality
warnings.filterwarnings("default", category=ImplicitModificationWarning)


ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"]



# 1) List all available keys
# list( ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"].keys() )
# e.g. ['RTCyT72_NoTx', 'RTCyPSCA_NoTx', 'CyPSCA_NoTx', 'CyT72_NoTx', … plus their reversals …]
# ['NoTx_RTCyT72',
#  'NoTx_RTCyPSCA',
#  'NoTx_CyPSCA',
#  'NoTx_CyT72',
#  'RTCyT72_RTCyPSCA',
#  'RTCyT72_CyPSCA',
#  'RTCyT72_CyT72',
#  'RTCyPSCA_CyPSCA',
#  'RTCyPSCA_CyT72',
#  'CyPSCA_CyT72']


# 2) Pick the one you want, e.g. "RTCyT72_NoTx"
result_dict = ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"]["NoTx_RTCyT72"]


result_dict


# 3) Inspect what’s inside
result_dict.keys()
# You should see something like: dict_keys(['enrichment', 'pvalues'])

# 4) Grab the p-values matrix
pval_df = result_dict["pvalues"]

# 5) (Optionally) Grab the ΔE values matrix
deltaE_df = result_dict["enrichment"]


import numpy as np

# Suppose you want only |ΔE|>0.1 AND p<0.05
deltaE = ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"]["NoTx_RTCyT72"]["enrichment"]
pmat   = ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"]["NoTx_RTCyT72"]["pvalue"]

mask = (np.abs(deltaE) > 0.1) & (pmat < 0.05)
# Create a masked array so that “non-significant” cells become white
display_matrix = deltaE.where(mask, other=0.0)

# Then plot display_matrix with seaborn or matplotlib, using the same colormap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(
    display_matrix,
    cmap="RdPu_r",                # or whichever diverging map you prefer
    vmin=-0.6, vmax=+0.6,         # fix your color limits to match all conditions
    annot=True, fmt=".2f",        # optional: print numbers
    cbar_kws={"label":"ΔEnrichment (RTCyT72 – NoTx)"}
)
plt.title("Significant Neighborhood Changes (|ΔE|>0.1 & p<0.05)")



# 1) List all available keys
# list( ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"].keys() )
# e.g. ['RTCyT72_NoTx', 'RTCyPSCA_NoTx', 'CyPSCA_NoTx', 'CyT72_NoTx', … plus their reversals …]
# ['NoTx_RTCyT72',
#  'NoTx_RTCyPSCA',
#  'NoTx_CyPSCA',
#  'NoTx_CyT72',
#  'RTCyT72_RTCyPSCA',
#  'RTCyT72_CyPSCA',
#  'RTCyT72_CyT72',
#  'RTCyPSCA_CyPSCA',
#  'RTCyPSCA_CyT72',
#  'CyPSCA_CyT72']

# 1) List the four comparisons you care about. These must match exactly the keys 
#    that appear under ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"].
#    In your case, those keys are named "NoTx_RTCyT72", "NoTx_RTCyPSCA", "NoTx_CyPSCA", "NoTx_CyT72".
comparisons = [
    "NoTx_RTCyT72",
    "NoTx_RTCyPSCA",
    "NoTx_CyPSCA",
    "NoTx_CyT72"
]

# 2) Pull out the diff_nhood_enrichment dictionary:
diff_dict = ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"]

# 3) Figure out the cluster ordering used by CellCharter,
#    so that our rows/columns line up with the ΔE plots exactly.
clusters = ST_sample.obs["cluster_cellcharter"].cat.categories.tolist()

# 4) Build a list of p-value DataFrames, in the same order as `comparisons`.
pval_matrices = []
for key in comparisons:
    if key not in diff_dict:
        raise KeyError(f"Could not find key '{key}' under diff_nhood_enrichment; available keys are:\n{list(diff_dict.keys())}")
    # Each entry (diff_dict[key]) is a dict with keys "enrichment" and "pvalues".
    pvals_df = diff_dict[key]["pvalue"].copy()
    # Reindex so that rows + columns follow the same cluster order
    pvals_df = pvals_df.reindex(index=clusters, columns=clusters)
    pval_matrices.append(pvals_df)

# 5) Plot them side by side in a 1×4 grid (or whatever layout you like).
n = len(comparisons)
ncols = 4
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(4*ncols, 4*nrows),
    tight_layout=True
)

# If there is only one row, ensure `axes` is 2D so we can index it as [row,col].
if nrows == 1:
    axes = np.array(axes).reshape(1, -1)

for idx, key in enumerate(comparisons):
    row = idx // ncols
    col = idx % ncols
    ax = axes[row, col]
    
    # Plot the p-value heatmap
    sns.heatmap(
        pval_matrices[idx],
        vmin=0.0, vmax=1.0,
        cmap="viridis_r",    # low p‐values = dark purple; high p‐values = light yellow
        square=True,
        cbar_kws={"label": "Permutation p-value"},
        ax=ax
    )
    # Make a title like “RTCyT72 vs NoTx (p-values)” from the key “NoTx_RTCyT72”
    # If you’d rather see “RTCyT72 vs NoTx” instead of “NoTx vs RTCyT72”, you can swap the pieces:
    cond1, cond2 = key.split("_")
    ax.set_title(f"{cond2} vs {cond1}\n(p-values)")
    ax.set_xlabel("Target cluster $C_j$")
    ax.set_ylabel("Source cluster $C_i$")

# If there are any “extra” subplots in the grid, delete them:
total_plots = nrows * ncols
for extra_idx in range(n, total_plots):
    r = extra_idx // ncols
    c = extra_idx % ncols
    fig.delaxes(axes[r, c])

plt.show()



# Suppose `pvals_df` is your (cluster × cluster) DataFrame of p-values,
# indexed and columned by the same “clusters” list we used before.

# 1) Make a masked copy where any p > 0.05 becomes NaN
masked_p = pvals_df.copy()
masked_p[masked_p > 0.05] = np.nan

# 2) Grab the reversed-viridis colormap and tell it to paint NaNs as lightgrey
cmap = plt.cm.viridis_r  # reversed viridis
cmap.set_bad(color="lightgrey")

# 3) Plot with seaborn, using mask=masked_p.isna() so that seaborn knows where those NaNs live
plt.figure(figsize=(5, 4))
sns.heatmap(
    masked_p,
    cmap=cmap,
    vmin=0.0, vmax=0.05,           # restrict color‐scaling to [0, 0.05]
    mask=masked_p.isna(),         # draw NaNs as “bad” (i.e. grey)
    square=True,
    cbar_kws={"label": "p-value (only 0–0.05)"},
    linewidths=0.5,
    linecolor="white"
)
plt.title("RTCyT72 vs NoTx (p ≤ 0.05); all others in grey")
plt.xlabel("Target cluster $C_j$")
plt.ylabel("Source cluster $C_i$")
plt.show()






ST_sample


ST_sample.obsm['spatial'].shape


def summarize_anndata(adata):
    report = []

    report.append(f"AnnData object with shape: {adata.shape}\n")

    report.append("OBSERVATIONS (.obs):")
    report.append(f"  Columns: {list(adata.obs.columns)}\n")

    report.append("VARIABLES (.var):")
    report.append(f"  Columns: {list(adata.var.columns)}\n")

    report.append("OBS MATRICES (.obsm):")
    for k in adata.obsm.keys():
        report.append(f"  {k}: shape {adata.obsm[k].shape}")

    report.append("\nOBS PAIRWISE (.obsp):")
    for k in adata.obsp.keys():
        report.append(f"  {k}: shape {adata.obsp[k].shape}")

    report.append("\nLAYERS:")
    for k in adata.layers.keys():
        report.append(f"  {k}: shape {adata.layers[k].shape}")

    report.append("\nUNSTRUCTURED (.uns):")
    for k in adata.uns_keys():
        item = adata.uns[k]
        if isinstance(item, dict):
            report.append(f"  {k}: dict with keys {list(item.keys())}")
        elif hasattr(item, 'shape'):
            report.append(f"  {k}: array-like, shape {item.shape}")
        else:
            report.append(f"  {k}: type {type(item)}")

    return "\n".join(report)

print(summarize_anndata(ST_sample))



ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"]


diff = ST_sample.uns["cluster_cellcharter_condition_diff_nhood_enrichment"]

for comp, comp_dict in diff.items():
    for mat in ("enrichment", "pvalue"):
        df = comp_dict[mat]
        # make row & column names strings
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)


del ST_sample.uns["shape_cluster_cellcharter"]


ST_sample.write("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/CellCharterAnalysis/onefer/onefer_enriched.h5ad")





import scanpy as sc

# 1) Run DE test for all clusters (one‐vs‐all)
sc.tl.rank_genes_groups(
    ST_sample,
    groupby="cluster_cellcharter",
    method="wilcoxon",
    n_genes=ST_sample.shape[1],  # test every gene
    key_added="rank_genes"
)

# 2) Extract the top 20 genes per cluster into a dictionary
top20_per_cluster = {}
for cluster in ST_sample.obs["cluster_cellcharter"].cat.categories:
    # cluster might be an integer (0,1,2…), so convert to string:
    cluster_str = str(cluster)
    df = sc.get.rank_genes_groups_df(ST_sample, group=cluster_str, key="rank_genes")
    # Sort by descending log‐fold‐change and take top 20
    df_sorted = df.sort_values("logfoldchanges", ascending=False)
    top20_per_cluster[cluster_str] = df_sorted["names"].tolist()[:20]


# 3) Print results
for cluster_str, genes in top20_per_cluster.items():
    print(f"\nCluster {cluster_str}:")
    print(", ".join(genes[0:10]))


top20_per_cluster


out_EA = base_folder / "figures" / "enrichment_analysis"


out_EA


for mouse in ST_sample.obs['mouse'].unique():
    print(mouse)


import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import gseapy as gp
from pathlib import Path

# ─── Step 0: Define output directory ────────────────────────────────────────────
# (Assumes you have already set `base_folder` somewhere above)
out_EA = base_folder / "figures" / "enrichment_analysis"
out_EA.mkdir(parents=True, exist_ok=True)

# ─── Step 1: Run DE test (one‐vs‐all) for every gene in each cluster ─────────────
sc.tl.rank_genes_groups(
    ST_sample,
    groupby="cluster_cellcharter",
    method="wilcoxon",
    n_genes=ST_sample.shape[1],  # test every gene
    key_added="rank_genes"
)

# ─── Parameters for Enrichment ─────────────────────────────────────────────────
n_pathways = 20
gene_sets = ["GO_Biological_Process_2021"]
organism = "Mouse"  # adjust if necessary

# ─── Step 2 & 3: For each cluster, extract all significant genes (adj p < 0.05)
#                and run Enrichr, then plot+save top n_pathways ─────────────────
for cluster in ST_sample.obs["cluster_cellcharter"].cat.categories:
    cluster_str = str(cluster)
    # 2a) Get DE results for this cluster
    df = sc.get.rank_genes_groups_df(
        ST_sample,
        group=cluster_str,
        key="rank_genes"
    )
    # 2b) Filter by adjusted p‐value < 0.05 (you can adjust threshold as needed)
    sig_df = df[df["pvals_adj"] < 0.05]
    gene_list = sig_df["names"].tolist()
    if len(gene_list) == 0:
        # Skip if no genes pass the threshold
        print(f"No significant genes (adj p < 0.05) found for cluster {cluster_str}. Skipping enrichment.")
        continue

    # 3a) Run Enrichr
    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism=organism,
        outdir=None  # don’t write any files to disk automatically
    )

    # 3b) Pull results, sort by adjusted p‐value, take top n_pathways
    res = enr.results.sort_values("Adjusted P-value").head(n_pathways)
    if res.shape[0] == 0:
        print(f"No pathways returned for cluster {cluster_str}.")
        continue

    # 3c) Compute Gene_Ratio and –log10(adj p)
    res["Gene_Ratio"] = res["Overlap"].apply(
        lambda x: int(x.split("/")[0]) / int(x.split("/")[1])
    )
    res["neg_log10_padj"] = -np.log10(res["Adjusted P-value"])

    # 3d) Sort by neg_log10_padj ascending for plotting
    res_plot = res.sort_values("neg_log10_padj", ascending=True)

    # 3e) Create scatter plot
    fig, ax = plt.subplots(figsize=(10, n_pathways * 0.4))
    sc_plot = ax.scatter(
        res_plot["Gene_Ratio"],
        res_plot["Term"],
        s=res_plot["neg_log10_padj"] * 50,
        c=res_plot["neg_log10_padj"],
        cmap="viridis"
    )
    ax.set_xlabel("Gene Ratio")
    ax.set_ylabel("")  # Terms are on the y‐axis
    ax.set_title(f"Top {n_pathways} Enriched Pathways for Cluster {cluster_str}")
    cbar = plt.colorbar(sc_plot, ax=ax)
    cbar.set_label("-log10(Adjusted P-value)")

    plt.tight_layout()
    # 3f) Save figure
    out_file = out_EA / f"cluster_{cluster_str}_enrichment.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved enrichment plot for cluster {cluster_str}: {out_file}")






# Calculating boundaries


cc.gr.connected_components(
    ST_sample,
    cluster_key="cluster_cellcharter",
    connectivity_key="spatial", # Again it adds "_connectivities" for some reason
    min_cells=50,
    out_key='component', # this is default, but I'm adding it to know the names
    copy=False
)


#   • ST_sample.obs["component"]  exists (one integer per cell).
#   • ST_sample.uns["shape_cluster_cellcharter"]["boundary"]  holds each component’s Polygon.

cc.tl.boundaries(
    ST_sample,
    cluster_key='cluster_cellcharter',
)


cc.tl.curl(
    ST_sample,
    cluster_key='cluster_cellcharter',
    copy=False
)
cc.tl.linearity(
    ST_sample,
    cluster_key='cluster_cellcharter',
    copy=False
)
cc.tl.elongation(
    ST_sample,
    cluster_key='cluster_cellcharter',
    copy=False
)

cc.tl.purity(
    ST_sample,
    cluster_key='cluster_cellcharter',
    library_key='mouse',
    copy=False
)


ST_sample.obs['mouse'].unique()


subset = ST_sample[ST_sample.obs["mouse"].isin(["RTCyPSCA_1_4", "NoTx_2_2"])]
print("Subset shape:", subset.shape)
print("cluster_cellcharter column exists:", "cluster_cellcharter" in subset.obs.columns)
print("Missing cluster labels:", subset.obs["cluster_cellcharter"].isna().sum())
print("Cluster value counts:\n", subset.obs["cluster_cellcharter"].value_counts())



ST_sample.obs["_dummy_cluster_key"] = ST_sample.obs["cluster_cellcharter"]

cc.pl.shape_metrics(
    ST_sample,
    condition_key="mouse",
    condition_groups=["RTCyPSCA_1_4", "NoTx_2_2"],
    component_key="cluster_cellcharter",
    cluster_key="_dummy_cluster_key",  # avoids the duplicate-column crash
    metrics=("curl", "linearity"),
    figsize=(6, 5),
    title="Curl & Linearity: RTCyT72 vs NoTx"
)






ST_sample.uns["shape_cluster_cellcharter"]["boundary"]


cc.tl.boundaries(ST_sample, cluster_key="cluster_cellcharter")


cc.pl.boundaries(
    ST_sample,
    sample='CyPSCA_1_1',
    library_key='mouse',
    component_key='cluster_cellcharter',
    show_cells=False
)


import cellcharter.pl._shape as ccshape
import cellcharter.pl as ccpl
import anndata as ad
import numpy as np
import geopandas
import matplotlib.pyplot as plt

def safe_boundaries(adata, sample, library_key, component_key, alpha_boundary=0.05, show_cells=True, save=None):
    adata = adata[adata.obs[library_key] == sample].copy()
    del adata.raw
    clusters = adata.obs[component_key].unique()

    # Pull only boundaries for visible clusters
    boundaries = {
        cluster: boundary
        for cluster, boundary in adata.uns[f"shape_{component_key}"]["boundary"].items()
        if cluster in clusters
    }

    gdf = geopandas.GeoDataFrame(geometry=list(boundaries.values()), index=np.arange(len(boundaries)).astype(str))

    # Skips problem if we don’t need cell plotting
    if not show_cells:
        gdf.plot(facecolor="none", edgecolor="black", linewidth=1)
        plt.axis("equal")
        if save:
            plt.savefig(save, dpi=300)
        plt.show()
        return

    # Otherwise, fallback to the original method
    ccshape.boundaries(
        adata, sample=sample, library_key=library_key, component_key=component_key,
        alpha_boundary=alpha_boundary, show_cells=True, save=save
    )

# Override the broken one
ccpl.boundaries = safe_boundaries



cc.pl.boundaries(
    ST_sample,
    sample='CyPSCA_1_1',
    library_key='mouse',
    component_key='cluster_cellcharter',
    show_cells=False
)



ST_sample.obs["instance_id"] = ST_sample.obs_names


ST_sample.obs.columns


ST_sample.obsp


cc.pl.boundaries(
    ST_sample,
    sample='CyPSCA_1_1',
    library_key='mouse',
    component_key='cluster_cellcharter',
    show_cells=False
)



subset = ST_sample[ST_sample.obs["sample_id"] == "CyPSCA_1_1"]

# 1. Confirm there are actually cells
print("Number of observations:", subset.n_obs)

# 2. Check if spatial coordinates exist
print("obsm keys:", subset.obsm.keys())

# 3. Print spatial shape and first few rows
if "spatial" in subset.obsm:
    spatial = subset.obsm["spatial"]
    print("Spatial shape:", spatial.shape)
    print("Any NaNs in spatial:", np.isnan(spatial).any())
    print("First few coordinates:\n", spatial[:5])
else:
    print("No 'spatial' in obsm")



# Compute component labels + shape metrics

cc.tl.curl(ST_sample,
          cluster_key='cluster_cellcharter',
          out_key='curl')





adata_dummy = ad.AnnData(X=np.array([['polygon1'], ['polygon2']], dtype=object))


adata_dummy.obs["region"]=["clusters", "cells"]


adata_dummy.obs['instance_id'] = ['0', '1']


ad.concat((ST_sample, adata_dummy))
