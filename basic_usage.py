# %%
import pandas as pd
import random
import preprocessing as pp
import utils 
import analysis
random.seed(13) # lucky number

#%%
# Load the data
small = pd.read_csv("../_datasets/7_screens/7screens_11kgenes.csv", index_col=0)
depmap = pd.read_csv("../_datasets/depmap/25Q2/gene_effect.csv", index_col=0)

depmap = depmap.iloc[:500]
small = small.iloc[:500]

#%%
inputs = {
    # First dataset is the REFERENCE dataset
    "DepMap (REFERENCE)": {
        "path": depmap, 
        "sort": "high",
    },
    
    # second dataset is the FOCUSED/TARGET dataset
    "7 screens (TARGET)": {
        "path": small,
        "sort": "high",
    }
}

# we need to drop NA values for matrix-wise operations otherwise it will take too long to run.
utils.initialize({
    "preprocessing": {
        "drop_na": False,
    }
})

data, genes = pp.load_datasets(inputs)

#%%
# In this part if you want normalization or mean-centering, you can do it here.
# This is optional and depends on your data.

# data['DepMap (REFERENCE)'] = pp.norm_and_center(data['DepMap (REFERENCE)'])


#%%
# this returns merged dataset with all samples
merged, reference_n_sample = pp.merge_datasets()

#%%
# we separate the reference and target datasets
# and get the contribution of each dataset
ref_contr, target_contr = analysis.get_contributions(merged, sample_size_separation=reference_n_sample)

#%%
# Formula 2: differential correlation (rM - rR) and direction matrix
diff_corr, direction = analysis.get_differential_correlation(
    merged, sample_size_separation=reference_n_sample
)

#%%
# Formula 3: covariation contributions normalized by group sample size
ref_cov_contr, target_cov_contr = analysis.get_covariation_contributions(
    merged, sample_size_separation=reference_n_sample
)

#%%
# Final score: weight a contribution matrix by corr(M)
target_final_score = analysis.get_final_score(target_contr, merged)

#%%
# just to validate 1 pair of genes
pair_r, pair_ref_contr, pair_target_contr = analysis.pairwise_contribution(
    merged.loc["ACP7", :], merged.loc["ACOT9", :], reference_n_sample
)

# %%
