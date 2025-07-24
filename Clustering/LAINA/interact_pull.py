def pull_interactions(ST_sample, sample_name, li):
    adata = ST_sample[ST_sample.obs['mouse']==sample_name].copy()
    
    #How to Identify neighbors
    li.ut.spatial_neighbors(adata, bandwidth=40, cutoff = 0.1, kernel='gaussian', set_diag=True)

    # Identify statistically relevant interactions
    lrdata = li.mt.bivariate(adata,
                resource_name='mouseconsensus',
                local_name='cosine', # Name of the function
                global_name="morans", # Name global function
                n_perms=100, # Number of permutations to calculate a p-value
                mask_negatives=False, # Whether to mask LowLow/NegativeNegative interactions
                add_categories=True, # Whether to add local categories to the results
                nz_prop=0.05, # Minimum expr. proportion for ligands/receptors and their subunits
                use_raw=False,
                verbose=True
                )

        # Add metadata columns
    condition = adata.obs['condition'].unique()
    mouse = adata.obs['mouse'].unique()

    if len(condition) != 1 or len(mouse) != 1:
        raise ValueError("Expected a single condition and mouse label in the filtered sample.")

    top_interactions = (
        lrdata.var[lrdata.var['morans_pvals'] <= 0.05]
        .sort_values("mean", ascending=False)
        .copy()
        .reset_index()
        .rename(columns={"index": "interactions"})
        .assign(condition=condition[0],
                mouse=mouse[0]
        )
    )
    
    return top_interactions