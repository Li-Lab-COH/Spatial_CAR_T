"""
Shared utilities for Getis-Ord Gi* hotspot analysis of spatial pathway scores.

These functions were originally defined inline in Getis_Ord_Gi_star_test.ipynb.
They are extracted here so the comparison notebook
(Hotspot_Abscopal_Comparison.ipynb) and the original notebook can share one
implementation instead of drifting copies.

Contents
--------
Coordinate handling
    get_tissue_scale_to_hires : cell-coords -> hires H&E scale factor
    get_tissue_coords_um      : tissue mask + coordinates in microns

Gi* statistic
    calc_getis_ord_gi_star    : fast Gi* with binary k-NN weights (self included)
    label_hotspot_components  : connected hotspot "islands" via a radius graph

Comparison helpers (new)
    hotspot_celltype_enrichment : density-normalized cell-type enrichment in/near
                                  hotspots, with a label-permutation null
    cliffs_delta                : nonparametric effect size for two groups

Notes
-----
Unlike the original notebook, the coordinate functions take their configuration
(tissue column, microns-per-pixel, hires scaling) as explicit arguments instead
of reading module-level globals, so the module has no hidden state.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

try:  # spatialdata is only needed for the coordinate helpers
    from spatialdata.transformations import get_transformation
except Exception:  # pragma: no cover - keeps the Gi* functions importable without spatialdata
    get_transformation = None


# ---------------------------------------------------------------------------
# Coordinate handling
# ---------------------------------------------------------------------------
def get_tissue_scale_to_hires(
    sdata,
    tissue_name,
    coordinate_system="downscale_to_hires",
    shapes_suffix="_cell_boundaries",
):
    """
    Returns the scale used to map cell coordinates to the hires H&E image.
    Translation is ignored because translation does not affect distances.
    """
    shapes_key = f"{tissue_name}{shapes_suffix}"

    if get_transformation is None:
        print("WARNING: spatialdata not available. Using scale=1.")
        return 1.0

    if shapes_key not in sdata.shapes:
        print(f"WARNING: {shapes_key} not found in sdata.shapes. Using scale=1.")
        return 1.0

    try:
        shapes_tf = get_transformation(sdata.shapes[shapes_key], get_all=True)[coordinate_system]
        return float(shapes_tf.scale[0])
    except Exception as e:
        print(f"WARNING: Could not get hires scale for {tissue_name}. Using scale=1. Error: {e}")
        return 1.0


def get_tissue_coords_um(
    adata,
    sdata,
    tissue_name,
    tissue_col,
    microns_per_pixel,
    apply_hires_scale=True,
):
    """
    Returns
        tissue_mask    : boolean mask for cells in this tissue
        coords_um      : x/y coordinates in microns
        scale_to_hires : scale applied before micron conversion

    Coordinate conversion:
        adata.obsm["spatial"] -> H&E pixels -> microns
        coords_um = adata.obsm["spatial"] * scale_to_hires * microns_per_pixel
    """
    tissue_mask = adata.obs[tissue_col].astype(str).eq(str(tissue_name)).values
    coords = np.asarray(adata.obsm["spatial"][tissue_mask], dtype=float)

    scale_to_hires = 1.0
    if apply_hires_scale:
        scale_to_hires = get_tissue_scale_to_hires(sdata, tissue_name)
        coords = coords * scale_to_hires

    coords_um = coords * microns_per_pixel
    return tissue_mask, coords_um, scale_to_hires


# ---------------------------------------------------------------------------
# Gi* statistic
# ---------------------------------------------------------------------------
def calc_getis_ord_gi_star(coords_um, values, k=50):
    """
    Fast Getis-Ord Gi* using binary k-nearest-neighbor weights.
    Includes self in the neighborhood, so this is Gi* rather than Gi.

    Positive z-score = local high-value clustering.
    Negative z-score = local low-value clustering.

    Returns
        z      : Gi* z-score per cell (standard normal under the null)
        q_hot  : BH-FDR q-value for the upper-tail (hot) test
        q_cold : BH-FDR q-value for the lower-tail (cold) test
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < k + 2:
        raise ValueError(f"Not enough cells for k={k}: n={n}")

    if np.nanstd(values) == 0:
        z = np.full(n, np.nan)
        q_hot = np.full(n, np.nan)
        q_cold = np.full(n, np.nan)
        return z, q_hot, q_cold

    # k + 1 because self is included
    n_neighbors = min(k + 1, n)

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(coords_um)
    _, idx = nn.kneighbors(coords_um)

    local_sum = values[idx].sum(axis=1)

    xbar = values.mean()
    s = values.std(ddof=0)

    sum_w = idx.shape[1]
    sum_w2 = sum_w  # binary weights: w_ij^2 == w_ij

    denom = s * np.sqrt((n * sum_w2 - sum_w**2) / (n - 1))
    z = (local_sum - xbar * sum_w) / denom

    p_hot = norm.sf(z)
    p_cold = norm.cdf(z)

    q_hot = multipletests(p_hot, method="fdr_bh")[1]
    q_cold = multipletests(p_cold, method="fdr_bh")[1]

    return z, q_hot, q_cold


def label_hotspot_components(coords_um, hotspot_mask, radius_um=50, min_cells=10):
    """
    Labels separate hotspot islands using a radius-neighbor graph.
    Returns -1 for non-hotspot cells or tiny hotspot components.
    """
    labels_full = np.full(len(hotspot_mask), -1, dtype=int)

    if hotspot_mask.sum() < min_cells:
        return labels_full

    hot_coords = coords_um[hotspot_mask]

    nn = NearestNeighbors(radius=radius_um)
    nn.fit(hot_coords)
    graph = nn.radius_neighbors_graph(hot_coords, mode="connectivity")

    n_components, component_labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    component_sizes = pd.Series(component_labels).value_counts().to_dict()

    kept_component_labels = np.array([
        lab if component_sizes.get(lab, 0) >= min_cells else -1
        for lab in component_labels
    ])

    labels_full[hotspot_mask] = kept_component_labels
    return labels_full


# ---------------------------------------------------------------------------
# Comparison helpers (new)
# ---------------------------------------------------------------------------
def hotspot_celltype_enrichment(
    coords_um,
    hotspot_mask,
    celltype_mask,
    radius_um=50,
    n_perm=1000,
    rng=None,
):
    """
    Density-normalized enrichment of a cell type within/near pathway hotspots.

    near-zone
        cells within ``radius_um`` of ANY hotspot cell. Hotspot cells are
        themselves in the near-zone (distance 0).

    enrichment
        log2( fraction of near-zone cells that are ``celltype``
              / global fraction of cells that are ``celltype`` ).
        Positive  -> the cell type is over-represented around hotspots.

    null
        The cell-type label is randomly re-assigned across all cells while the
        positions (and therefore the near-zone) stay fixed. This is the
        complete-spatial-randomness-of-labels null and is identical to drawing
        ``n_near`` cells without replacement; it is sampled with a hypergeometric
        draw for speed (equivalent to a label permutation, but O(n_perm) instead
        of O(n_perm * n_cells)). Density differences between tissues cancel,
        which is what makes the enrichment comparable across tissues.

    Returns a dict:
        n_hotspot, n_near, n_celltype,
        obs_frac, baseline_frac, log2_enrich,
        emp_p (two-sided), z, null_mean, null_std
    """
    coords_um = np.asarray(coords_um, dtype=float)
    hotspot_mask = np.asarray(hotspot_mask, dtype=bool)
    celltype_mask = np.asarray(celltype_mask, dtype=bool)
    n = len(hotspot_mask)
    rng = np.random.default_rng(rng)

    result = {
        "n_hotspot": int(hotspot_mask.sum()),
        "n_near": 0,
        "n_celltype": int(celltype_mask.sum()),
        "obs_frac": np.nan,
        "baseline_frac": np.nan,
        "log2_enrich": np.nan,
        "emp_p": np.nan,
        "z": np.nan,
        "null_mean": np.nan,
        "null_std": np.nan,
    }

    baseline = celltype_mask.mean() if n else np.nan
    result["baseline_frac"] = float(baseline) if np.isfinite(baseline) else np.nan

    n_ct = int(celltype_mask.sum())
    if hotspot_mask.sum() == 0 or n_ct == 0:
        return result

    # near-zone: any cell whose nearest hotspot cell is within radius_um.
    # Using the single nearest-hotspot distance is equivalent to "within
    # radius of any hotspot cell" but far cheaper than materializing full
    # radius-neighbor lists for every cell in a large tissue.
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(coords_um[hotspot_mask])
    nearest_hot_dist, _ = nn.kneighbors(coords_um)
    near_mask = nearest_hot_dist.ravel() <= radius_um
    n_near = int(near_mask.sum())
    result["n_near"] = n_near
    if n_near == 0:
        return result

    obs_frac = float(celltype_mask[near_mask].mean())
    result["obs_frac"] = obs_frac
    result["log2_enrich"] = float(np.log2(obs_frac / baseline)) if obs_frac > 0 else -np.inf

    # null: count of celltype cells among n_near cells drawn without replacement
    null_counts = rng.hypergeometric(ngood=n_ct, nbad=n - n_ct, nsample=n_near, size=n_perm)
    null_fracs = null_counts / n_near
    null_mean = float(null_fracs.mean())
    null_std = float(null_fracs.std(ddof=1))
    result["null_mean"] = null_mean
    result["null_std"] = null_std
    result["z"] = float((obs_frac - null_mean) / null_std) if null_std > 0 else np.nan

    diff_obs = abs(obs_frac - null_mean)
    result["emp_p"] = float((np.sum(np.abs(null_fracs - null_mean) >= diff_obs) + 1) / (n_perm + 1))
    return result


def cliffs_delta(a, b):
    """
    Cliff's delta effect size for two samples: (#a>b - #a<b) / (len(a)*len(b)).
    +1 means every value in ``a`` exceeds every value in ``b``; -1 the reverse;
    0 means full overlap. NaNs are dropped. Returns np.nan if either side is empty.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.sum() / (len(a) * len(b)))
