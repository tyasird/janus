#%%
import random
import pandas as pd
from janusbio import preprocessing as pp
from janusbio import utils
from janusbio.analysis import get_contributions
from janusbio.helpers import binary, perform_corr, quick_sort
from janusbio.permutation_tests import (
    adaptive_topk_rank_permutation_test,
    adaptive_topk_rank_permutation_test_parallel,
)
random.seed(13)  # lucky number

# %%
# Load the data
experiment = pd.read_csv("../../_datasets/janus_data_from_thomas/mito_screens_z_normalized_for_janus.csv").T
reference = pd.read_csv("../../_datasets/janus_data_from_thomas/DM23Q2_z_normalized_for_janus.csv").T

experiment_corr = perform_corr(experiment, "numpy")
reference_corr = perform_corr(reference, "numpy")

# %%
inputs = {
    # First dataset is the REFERENCE dataset
    "DepMap (REFERENCE)": {
        "path": reference,
        "sort": "high",
    },
    # second dataset is the EXPERIMENT dataset
    "7 screens (EXPERIMENT)": {
        "path": experiment,
        "sort": "high",
    }
}

# we need to drop NA values for matrix-wise operations 
# otherwise it will take too long to run.
utils.initialize({
    "preprocessing": {
        "drop_na": True, #we lose 161 genes from depmap
    }
})

data, genes = pp.load_datasets(inputs)
merged, reference_sample_size = pp.merge_datasets()

# %%
# we separate the reference and experiment datasets
# and get the contribution of each dataset
reference_contr, experiment_contr = get_contributions(merged, reference_sample_size=reference_sample_size)

# %%


# # just to validate 1 pair of genes
# pair_r, pair_reference_contr, pair_experiment_contr = analysis.pairwise_contribution(
#     merged.loc["ACP7", :], merged.loc["ACOT9", :], reference_sample_size
# )


# #%%
# # Formula 2: differential correlation (rM - rR) and direction matrix
# diff_corr, direction = analysis.get_differential_correlation(
#     merged, reference_sample_size=reference_sample_size
# )

# #%%
# # Formula 3: covariation contributions normalized by group sample size
# reference_cov_contr, experiment_cov_contr = analysis.get_covariation_contributions(
#     merged, reference_sample_size=reference_sample_size
# )

# #%%
# # Final score: weight a contribution matrix by corr(M)
# experiment_final_score = analysis.get_final_score(experiment_contr, merged)


# # Output folder for adaptive top-k outputs
# output_folder = utils.dload("config").get("output_folder", "output")
# os.makedirs(output_folder, exist_ok=True)
# selected_samples_txt_path = os.path.join(output_folder, "adaptive_topk_selected_samples.txt")


# %%
# adaptive_topk_rank_results = adaptive_topk_rank_permutation_test(
#     merged_df=merged,
#     reference_sample_size=reference_sample_size,
#     n_experiment=None,
#     create_table_with_perm_values=False,
#     top_k=10000,
#     n_perm=10000,
#     random_state=13,
#     sanity_check=True,
# )

# Parallel version example:
adaptive_topk_rank_results = adaptive_topk_rank_permutation_test_parallel(
    merged_df=merged,
    reference_sample_size=reference_sample_size,
    n_experiment=None,
    create_table_with_perm_values=False,
    create_table_with_selected_samples=True,
    top_k=10000,
    n_perm=10000,
    random_state=13,
    sanity_check=False,
    n_jobs=5,
    permutations_per_job=2000,
)


rank_pvalues = adaptive_topk_rank_results["rank_pvalues"]
rank_summary_pvalues = adaptive_topk_rank_results["summary_pvalues"]
selected_samples = adaptive_topk_rank_results["selected_samples"]

paper_rank_df = rank_pvalues.loc[
    rank_pvalues["rank"].isin([1, 100, 1000]),
    ["rank", "observed_value", "p_empirical", "n_perm_ge_observed"]
].copy()

paper_summary_df = rank_summary_pvalues[
    ["statistic", "observed", "p_empirical", "n_perm_ge_observed"]
].copy()
# %%
