# --- deps ---
import numpy as np
import pandas as pd
import scipy.sparse as sp

def annotate_macrophage_polarization(
    adata,
    *,
    counts_layer="counts",            # use "counts" layer if present; else X
    detect_threshold=2,               # per-gene detection threshold (≥1 UMI default)
    m1_total_req=2, m1_core_req=1,    # tiered rule for M1 call
    m2_total_req=2, m2_core_req=1,    # tiered rule for M2 call
    tie_margin=0.02,                  # score advantage to break ties
    write_prefix="",                  # optional prefix for obs column names
    verbose=True
):
    """
    Annotate macrophage polarization (M1/M2) and flag Spp1/Glp1r+ cells.

    Adds columns (with optional write_prefix):
      - M1_core_n, M1_acc_n, M2_core_n, M2_acc_n, M1_total, M2_total
      - M1_score_tiered, M2_score_tiered
      - Mac_polarization_tiered (Categorical: M1, M2, Both, Neither)
      - Mac_like_tiered (0/1)
      - Spp1_pos, Glp1r_pos (0/1; ≥1 UMI)
      - Mac_Spp1_pos, Mac_Glp1r_pos (among Mac_like_tiered==1)
      - Mac_marker_combo (Categorical for macrophage-like cells: None, Spp1, Glp1r, Both; others = Non-mac)
      
      To do:
      - Edit the genes for human samples
      - Mark macrophages first, then polarize
          
    """

    # --------------------------
    # Select count matrix
    # --------------------------
    X = adata.layers[counts_layer] if (counts_layer in adata.layers) else adata.X

    # --------------------------
    # Define gene sets
    # --------------------------
    m1_all = ['Ccl3','Ccl4','Ccl5','Cd40','Cxcl10','Cxcl9','Il1b','Il6','Irf1','Irf5','Nos2','Psmb9','Stat1','Tap1','Tnf', 'Cd80','Cd86','Ccr7']
    m2_all = ['Anxa1','Apoe','Arg1','Axl','C1qa','C1qb','C1qc','Chil3','Csf1r','Fn1','Gas6','Il10','Itgav','Itgb1','Lgals3','Lrp1','Mertk','Mrc1','Pros1','Retnla','Vegfa']

    m1_core = ['Nos2', 'Il1b', 'Tnf', 'Stat1', 'Irf1', 'Irf5', 'Cxcl9', 'Cxcl10', 'CD40', 'Psmb9', 'H2-Ab1']
    m1_acc  = sorted(list(set(m1_all) - set(m1_core)))

    m2_core = ['Mrc1','Arg1','Retnla','Chil3','Mertk','Axl','Csf1r','Il10', 'Tgfb1', 'Lgals3']
    m2_acc  = sorted(list(set(m2_all) - set(m2_core)))

    # --------------------------
    # Case-insensitive gene matching
    # --------------------------
    var_names = pd.Index(adata.var_names)
    lower_to_idx = {g.lower(): i for i, g in enumerate(var_names)}

    def get_idx_list(glist):
        idx = []
        missing = []
        for g in glist:
            i = lower_to_idx.get(g.lower(), None)
            if i is None: missing.append(g)
            else: idx.append(i)
        return idx, missing

    m1c_idx, m1c_missing = get_idx_list(m1_core)
    m1a_idx, m1a_missing = get_idx_list(m1_acc)
    m2c_idx, m2c_missing = get_idx_list(m2_core)
    m2a_idx, m2a_missing = get_idx_list(m2_acc)

    # --------------------------
    # Binary detection with threshold (≥ detect_threshold)
    #     Note: your earlier snippet used ( > 1 ) which equals ≥2 UMIs.
    #     Here we default to ≥1 UMI; set detect_threshold=2 to match the old behavior.
    # --------------------------
    def bin_detect(M):
        if sp.issparse(M): 
            return (M >= detect_threshold).astype(np.int8)
        return (M >= detect_threshold).astype(np.int8)

    X_m1c = bin_detect(X[:, m1c_idx]) if len(m1c_idx) else np.zeros((adata.n_obs, 0), dtype=np.int8)
    X_m1a = bin_detect(X[:, m1a_idx]) if len(m1a_idx) else np.zeros((adata.n_obs, 0), dtype=np.int8)
    X_m2c = bin_detect(X[:, m2c_idx]) if len(m2c_idx) else np.zeros((adata.n_obs, 0), dtype=np.int8)
    X_m2a = bin_detect(X[:, m2a_idx]) if len(m2a_idx) else np.zeros((adata.n_obs, 0), dtype=np.int8)

    # --------------------------
    # 5) Per-cell counts of detected genes
    # --------------------------
    def row_sum(A):
        return np.array(A.sum(axis=1)).ravel() if sp.issparse(A) else A.sum(axis=1)

    m1_core_n = row_sum(X_m1c)
    m1_acc_n  = row_sum(X_m1a)
    m2_core_n = row_sum(X_m2c)
    m2_acc_n  = row_sum(X_m2a)

    m1_total = m1_core_n + m1_acc_n
    m2_total = m2_core_n + m2_acc_n

    # --------------------------
    # 6) Tiered rules + scores
    # --------------------------
    m1_conf = (m1_total >= m1_total_req) & (m1_core_n >= m1_core_req)
    m2_conf = (m2_total >= m2_total_req) & (m2_core_n >= m2_core_req)

    m1_score = m1_total / max(1, (len(m1_core)+len(m1_acc)))
    m2_score = m2_total / max(1, (len(m2_core)+len(m2_acc)))

    label_tiered = np.array(["Neither"]*adata.n_obs, dtype=object)
    label_tiered[m1_conf & ~m2_conf] = "M1"
    label_tiered[m2_conf & ~m1_conf] = "M2"

    both = m1_conf & m2_conf
    m1_win = (m1_score[both] > m2_score[both] + tie_margin)
    m2_win = (m2_score[both] > m1_score[both] + tie_margin)

    tmp = np.array(["Both"]*both.sum(), dtype=object)
    tmp[m1_win] = "M1"
    tmp[m2_win] = "M2"
    label_tiered[both] = tmp

    # --------------------------
    # 7) Write outputs back to .obs
    # --------------------------
    p = write_prefix

    adata.obs[f"{p}M1_core_n"] = m1_core_n
    adata.obs[f"{p}M1_acc_n"]  = m1_acc_n
    adata.obs[f"{p}M2_core_n"] = m2_core_n
    adata.obs[f"{p}M2_acc_n"]  = m2_acc_n
    adata.obs[f"{p}M1_total"]  = m1_total
    adata.obs[f"{p}M2_total"]  = m2_total

    adata.obs[f"{p}M1_score_tiered"] = m1_score
    adata.obs[f"{p}M2_score_tiered"] = m2_score

    adata.obs[f"{p}Mac_polarization_tiered"] = pd.Categorical(
        label_tiered, categories=["M1","M2","Both","Neither"], ordered=False
    )
    adata.obs[f"{p}Mac_like_tiered"] = (adata.obs[f"{p}Mac_polarization_tiered"] != "Neither").astype(int)

    # --------------------------
    # 8) Marker positivity (≥1 UMI) for Spp1 and Glp1r
    # --------------------------
    def gene_pos(gene, thr=1):
        i = lower_to_idx.get(gene.lower(), None)
        if i is None:
            return np.zeros(adata.n_obs, dtype=np.int8), True
    
        # Use getcol for sparse matrices (avoids 2D slicing quirks), else normal slice
        if sp.issparse(X):
            col = X.getcol(i)            # (n_obs, 1) sparse column
            arr1d = np.asarray(col.toarray()).ravel()
        else:
            col = X[:, i]                # numpy array; could be (n_obs,) or (n_obs,1)
            arr1d = np.asarray(col).ravel()
    
        arr = (arr1d >= thr).astype(np.int8)
        return arr, False

    spp1_pos, spp1_missing = gene_pos("Spp1", thr=1)      # explicitly ≥1 UMI
    glp1r_pos, glp1r_missing = gene_pos("Glp1r", thr=1)   # explicitly ≥1 UMI

    adata.obs[f"{p}Spp1_pos"]  = spp1_pos
    adata.obs[f"{p}Glp1r_pos"] = glp1r_pos

    # Among macrophage-like cells, convenience flags
    mac_like = adata.obs[f"{p}Mac_like_tiered"].values.astype(bool)
    adata.obs[f"{p}Mac_Spp1_pos"]  = (mac_like & (spp1_pos == 1)).astype(int)
    adata.obs[f"{p}Mac_Glp1r_pos"] = (mac_like & (glp1r_pos == 1)).astype(int)

    # Combo categorical (for macrophage-like cells)
    combo = np.full(adata.n_obs, "Non-mac", dtype=object)
    combo_mac = np.where((spp1_pos==1) & (glp1r_pos==1), "Both",
                 np.where((spp1_pos==1), "Spp1",
                 np.where((glp1r_pos==1), "Glp1r", "None")))
    combo[mac_like] = combo_mac[mac_like]
    adata.obs[f"{p}Mac_marker_combo"] = pd.Categorical(
        combo, categories=["Non-mac","None","Spp1","Glp1r","Both"], ordered=False
    )

    # --------------------------
    # 9) Verbose summary
    # --------------------------
    if verbose:
        missing_report = {
            "m1_core_missing": m1c_missing, "m1_acc_missing": m1a_missing,
            "m2_core_missing": m2c_missing, "m2_acc_missing": m2a_missing,
            "Spp1_missing": ["Spp1"] if spp1_missing else [],
            "Glp1r_missing": ["Glp1r"] if glp1r_missing else []
        }
        print("Mac polarization added. N cells =", adata.n_obs)
        print("Class counts:", adata.obs[f"{p}Mac_polarization_tiered"].value_counts(dropna=False).to_dict())
        print("Spp1+ (≥1 UMI):", int(adata.obs[f"{p}Spp1_pos"].sum()),
              "| Glp1r+ (≥1 UMI):", int(adata.obs[f"{p}Glp1r_pos"].sum()))
        print("Mac marker combos:", adata.obs[f"{p}Mac_marker_combo"].value_counts(dropna=False).to_dict())

        has_missing = {k:v for k,v in missing_report.items() if len(v)>0}
        if len(has_missing):
            print("\n[Warning] Missing genes (not in var_names):")
            for k,v in has_missing.items():
                print(f"  - {k}: {', '.join(v)}")

    return adata  # modified in place; also returned for convenience
