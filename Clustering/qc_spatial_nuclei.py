# qc_spatial_nuclei.py
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from typing import Optional, Sequence
from anndata import AnnData

# ----------------------------
# Helpers
# ----------------------------
def _get_spatial_coords(adata: AnnData):
    if "spatial" in adata.obsm_keys():
        XY = adata.obsm["spatial"]
        if XY.shape[1] >= 2:
            return XY[:, 0], XY[:, 1]
    for cols in (("x","y"), ("X","Y")):
        if all(c in adata.obs.columns for c in cols):
            return adata.obs[cols[0]].values, adata.obs[cols[1]].values
    raise ValueError("No spatial coordinates found. Provide adata.obsm['spatial'] or obs columns x/y or X/Y.")

def _detect_area(adata: AnnData) -> Optional[np.ndarray]:
    for k in ("area","nucleus_area","Area","nuc_area"):
        if k in adata.obs.columns:
            return adata.obs[k].to_numpy()
    return None

def _ensure_counts_csr(adata: AnnData):
    X = adata.X
    if sp.issparse(X):
        return X.tocsr()
    # dense -> sparse for scalable ops
    return sp.csr_matrix(X)

def _add_basic_qc(adata: AnnData,
                  mito_prefixes: Sequence[str] = ("mt-", "MT-", "Mt-"),
                  ribo_prefixes: Sequence[str] = ("Rpl","Rps","RPL","RPS")):
    X = _ensure_counts_csr(adata)

    # total_counts / n_genes_by_counts if missing
    if "total_counts" not in adata.obs.columns:
        total = np.asarray(X.sum(axis=1)).ravel()
        adata.obs["total_counts"] = total
    if "n_genes_by_counts" not in adata.obs.columns:
        if sp.issparse(X):
            n_genes = np.diff(X.indptr)  # nonzeros per row
        else:
            n_genes = (X > 0).sum(axis=1)
        adata.obs["n_genes_by_counts"] = np.array(n_genes).ravel()

    # mito / ribo fractions
    var_names = adata.var_names.to_numpy()
    mito_mask = np.zeros(len(var_names), dtype=bool)
    for p in mito_prefixes:
        mito_mask |= np.char.startswith(var_names.astype(str), p)

    ribo_mask = np.zeros(len(var_names), dtype=bool)
    for p in ribo_prefixes:
        ribo_mask |= np.char.startswith(var_names.astype(str), p)

    mito_sum = np.asarray(X[:, mito_mask].sum(axis=1)).ravel() if mito_mask.any() else np.zeros(adata.n_obs)
    ribo_sum = np.asarray(X[:, ribo_mask].sum(axis=1)).ravel() if ribo_mask.any() else np.zeros(adata.n_obs)
    total = adata.obs["total_counts"].to_numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        adata.obs["pct_mt"] = np.where(total > 0, 100.0 * mito_sum / total, 0.0)
        adata.obs["pct_ribo"] = np.where(total > 0, 100.0 * ribo_sum / total, 0.0)

def _safe_group_levels(adata: AnnData, key: str) -> Optional[Sequence]:
    if key in adata.obs.columns:
        vals = adata.obs[key]
        if hasattr(vals, "cat"):
            return list(vals.cat.categories)
        return sorted(pd.unique(vals))
    return None

def _make_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def _basic_hist(series, title, xlabel, outpath):
    plt.figure(figsize=(6,4))
    x = pd.Series(series).dropna()
    plt.hist(x, bins=60)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _grouped_violin_df(df: pd.DataFrame, value: str, group: str, outpath: str):
    """
    Draw a violin plot for `value` grouped by `group` using the already-filtered df.
    Does not touch AnnData; safe for length-mismatched filtering.
    """
    if group not in df.columns or value not in df.columns:
        return
    g = df[[value, group]].dropna()
    if g.empty:
        return

    # Determine group order; keep categorical order if available
    if hasattr(g[group], "cat"):
        levels = list(g[group].cat.categories)
    else:
        levels = sorted(pd.unique(g[group]))

    # Build data and drop empty groups (can happen after filtering)
    data = []
    kept_levels = []
    for lvl in levels:
        arr = g.loc[g[group] == lvl, value].to_numpy()
        if arr.size > 0:
            data.append(arr)
            kept_levels.append(lvl)
    if not data:
        return

    plt.figure(figsize=(max(6, min(18, 0.4*len(kept_levels)+4)), 4.5))
    plt.violinplot(data, showmedians=True)
    plt.xticks(range(1, len(kept_levels)+1), [str(l) for l in kept_levels], rotation=90)
    plt.ylabel(value)
    plt.title(f"{value} by {group} (98th pct outliers removed per group)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _drop_group_outliers(adata: AnnData, value: str, group: str) -> pd.DataFrame:
    df = adata.obs[[value, group]].copy()
    if df[group].isnull().any():
        df = df.dropna(subset=[group])
    # compute 98th percentile within each group
    qs = df.groupby(group, observed=False)[value].quantile(0.98)
    keep = df.apply(lambda r: r[value] <= qs.loc[r[group]], axis=1)
    return df.loc[keep].copy()

def _spatial_scatter(x, y, c, title, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=3, alpha=0.8, c=c)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()  # histology coords
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _top_gene_dominance(adata: AnnData, top_k: int = 1) -> np.ndarray:
    X = _ensure_counts_csr(adata)
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    # For each cell, compute fraction in top_k expressed genes
    # Efficiently approximate by taking per-row max (top1). For k>1, we’d sort row segments.
    fractions = np.zeros(adata.n_obs, dtype=float)
    indptr = X.indptr
    data = X.data
    for i in range(adata.n_obs):
        start, end = indptr[i], indptr[i+1]
        if end > start:
            row = data[start:end]
            top = np.partition(row, -top_k)[-top_k:].sum()
            total = row.sum()
            fractions[i] = top / total if total > 0 else 0.0
        else:
            fractions[i] = 0.0
    return fractions

def _cumulative_capture_curve(total_counts: np.ndarray, outpath: str):
    # Lorenz-like curve: nuclei sorted by UMIs, cumulative nuclei vs cumulative UMIs
    x = np.asarray(total_counts).astype(float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0, None)
    if x.size == 0:
        return
    order = np.argsort(x)
    x_sorted = x[order]
    cum_x = np.cumsum(x_sorted)
    cum_x = cum_x / cum_x[-1] if cum_x[-1] > 0 else cum_x
    frac_cells = np.linspace(0,1,len(x_sorted), endpoint=True)
    plt.figure(figsize=(6,4))
    plt.plot(frac_cells, cum_x)
    plt.title("Cumulative UMI capture across nuclei")
    plt.xlabel("Fraction of nuclei (low→high UMI)")
    plt.ylabel("Fraction of total UMIs")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _complexity_curve(adata: AnnData, outpath: str, nbins: int = 30):
    # Median genes detected vs. total UMIs (binned)
    df = adata.obs[["total_counts","n_genes_by_counts"]].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return
    bins = np.quantile(df["total_counts"], np.linspace(0,1,nbins+1))
    bins = np.unique(bins)
    if len(bins) < 3:
        return
    labels = pd.IntervalIndex.from_breaks(bins, closed="right")
    cats = pd.cut(df["total_counts"], bins=bins, include_lowest=True)
    m = df.groupby(cats)["n_genes_by_counts"].median()
    centers = np.array([i.mid for i in m.index.categories])
    plt.figure(figsize=(6,4))
    plt.plot(centers, m.to_numpy(), marker="o")
    plt.title("Complexity curve: median genes vs. UMIs")
    plt.xlabel("Total UMIs (bin center)")
    plt.ylabel("Median genes detected")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _metric_correlations(adata: AnnData, outpath: str,
                         cols=("total_counts","n_genes_by_counts","pct_mt","pct_ribo")):
    use = [c for c in cols if c in adata.obs.columns]
    if len(use) < 2:
        return
    df = adata.obs[use].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return
    corr = df.corr(method="spearman")
    # simple heatmap w/ matplotlib only
    plt.figure(figsize=(4.5,4))
    im = plt.imshow(corr.values, aspect="equal")
    plt.xticks(range(len(use)), use, rotation=90)
    plt.yticks(range(len(use)), use)
    plt.title("Spearman correlations (QC metrics)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ----------------------------
# Main entry
# ----------------------------
def run_spatial_qc(
    adata: AnnData,
    outdir: str = "qc_output",
    groups: Sequence[str] = ("mouse","condition","sample_id"),
    mito_prefixes: Sequence[str] = ("mt-", "MT-", "Mt-"),
    ribo_prefixes: Sequence[str] = ("Rpl","Rps","RPL","RPS"),
    topk_dominance: int = 3
):
    """
    Extensible QC pipeline for nucleus-resolved spatial transcriptomics (StarDist segmented).
    Saves PNGs + summary CSV.
    """
    _make_outdir(outdir)
    _add_basic_qc(adata, mito_prefixes=mito_prefixes, ribo_prefixes=ribo_prefixes)

    # 1) Distributions (global)
    _basic_hist(adata.obs["total_counts"], "UMIs per nucleus", "Total UMIs", os.path.join(outdir,"hist_total_umis.png"))
    _basic_hist(adata.obs["n_genes_by_counts"], "Genes per nucleus", "Genes detected", os.path.join(outdir,"hist_genes.png"))
    _basic_hist(adata.obs["pct_mt"], "% mitochondrial", "% mito", os.path.join(outdir,"hist_pct_mito.png"))
    if "pct_ribo" in adata.obs.columns:
        _basic_hist(adata.obs["pct_ribo"], "% ribosomal", "% ribo", os.path.join(outdir,"hist_pct_ribo.png"))

    # 2) Grouped violins (98th pct outliers removed per group)
    for g in groups:
        if g in adata.obs.columns:
            for v in ("total_counts","n_genes_by_counts","pct_mt","pct_ribo"):
                if v in adata.obs.columns:
                    # build a filtered view for labelling but pass full adata for levels
                    df_f = _drop_group_outliers(adata, v, g)
                    # temporary adata-like object for plotting without rewriting API
                    tmp = adata.copy()
                    tmp.obs = df_f  # contains only v and g, but that's all we need
                    _grouped_violin(adata=adata, value=v, group=g,
                                    outpath=os.path.join(outdir, f"violin_{v}_by_{g}.png"))

    # 3) Spatial QC maps
    try:
        x, y = _get_spatial_coords(adata)
        for v, title in (
            ("total_counts","Spatial UMIs"),
            ("n_genes_by_counts","Spatial genes detected"),
            ("pct_mt","Spatial % mito"),
        ):
            if v in adata.obs.columns:
                _spatial_scatter(x, y, adata.obs[v].to_numpy(), title, os.path.join(outdir, f"spatial_{v}.png"))
    except Exception as e:
        # No spatial coordinates — skip gracefully
        pass

    # 4) Area vs UMIs (if area available)
    area = _detect_area(adata)
    if area is not None:
        plt.figure(figsize=(5,4))
        plt.scatter(area, adata.obs["total_counts"].to_numpy(), s=5, alpha=0.6)
        plt.xlabel("Nucleus area")
        plt.ylabel("Total UMIs")
        plt.title("Area vs UMIs")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,"area_vs_umis.png"), dpi=200)
        plt.close()

    # 5) Top-gene dominance
    frac_topk = _top_gene_dominance(adata, top_k=topk_dominance)
    adata.obs[f"frac_top{topk_dominance}_genes"] = frac_topk
    _basic_hist(frac_topk, f"Fraction in top {topk_dominance} genes", "Fraction of UMIs", os.path.join(outdir, f"hist_frac_top{topk_dominance}.png"))

    # 6) Cumulative capture curve
    _cumulative_capture_curve(adata.obs["total_counts"].to_numpy(), os.path.join(outdir,"curve_cumulative_capture.png"))

    # 7) Complexity curve (median genes vs UMIs binned)
    _complexity_curve(adata, os.path.join(outdir,"curve_complexity_genes_vs_umis.png"))

    # 8) Correlations
    _metric_correlations(adata, os.path.join(outdir,"corr_qc_metrics.png"))

    # 9) Per-group summary table (median/IQR)
    summaries = []
    metrics = ["total_counts","n_genes_by_counts","pct_mt","pct_ribo", f"frac_top{topk_dominance}_genes"]
    metrics = [m for m in metrics if m in adata.obs.columns]
    for g in groups:
        if g in adata.obs.columns:
            grp = adata.obs[[g] + metrics].copy()
            # robust summaries per group
            def iqr(x): 
                q1, q3 = np.nanpercentile(x, [25, 75])
                return q3 - q1
            agg = grp.groupby(g, observed=False).agg(
                **{f"{m}_median": (m, "median") for m in metrics},
                **{f"{m}_IQR": (m, iqr) for m in metrics},
                **{f"{m}_mean": (m, "mean") for m in metrics},
                **{f"{m}_n": (m, "count") for m in metrics},
            )
            agg["grouping"] = g
            summaries.append(agg.reset_index())
    if summaries:
        summary_df = pd.concat(summaries, axis=0, ignore_index=True)
        summary_df.to_csv(os.path.join(outdir, "qc_summary_by_group.csv"), index=False)

    # 10) Global summary
    global_summary = {
        "n_nuclei": adata.n_obs,
        "UMIs_median": float(np.nanmedian(adata.obs["total_counts"])),
        "UMIs_IQR": float(np.nanpercentile(adata.obs["total_counts"], 75) - np.nanpercentile(adata.obs["total_counts"], 25)),
        "genes_median": float(np.nanmedian(adata.obs["n_genes_by_counts"])),
        "genes_IQR": float(np.nanpercentile(adata.obs["n_genes_by_counts"], 75) - np.nanpercentile(adata.obs["n_genes_by_counts"], 25)),
        "pct_mt_median": float(np.nanmedian(adata.obs["pct_mt"])) if "pct_mt" in adata.obs.columns else np.nan,
    }
    pd.DataFrame([global_summary]).to_csv(os.path.join(outdir, "qc_summary_global.csv"), index=False)

    return outdir
