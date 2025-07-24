def pull_interactions(ST_sample, sample_name):
    adata = ST_sample[ST_sample.obs['mouse']==sample_name].copy()
    
    #How to Identify neighbors
    li.ut.spatial_neighbors(adata, bandwidth=40, cutoff = 0.1, kernel='gaussian', set_diag=True)

    # Identify statistically relevant interactions
    lrdata = li.mt.bivariate(adata,
                resource_name='mouseconsensus', # NOTE: uses HUMAN gene symbols!
                local_name='cosine', # Name of the function
                global_name="morans", # Name global function
                n_perms=100, # Number of permutations to calculate a p-value
                mask_negatives=False, # Whether to mask LowLow/NegativeNegative interactions
                add_categories=True, # Whether to add local categories to the results
                nz_prop=0.05, # Minimum expr. proportion for ligands/receptors and their subunits
                use_raw=False,
                verbose=True
                )

    top_interactions = lrdata.var[lrdata.var['morans_pvals'] <= 0.05].sort_values("mean", ascending=False).head(50).index

    return top_interactions