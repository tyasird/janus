import numpy as np
import pandas as pd

from .helpers import perform_corr


# --------------------------------------------------------------------------
# Contribution calculation functions
# --------------------------------------------------------------------------


# just for validation
def pairwise_contribution(x, y, reference_sample_size):
    # x = gene1 vector
    # y = gene2 vector
    # reference_sample_size separates reference and experiment samples

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Split at the boundary before masking to preserve block identity.
    x_reference_raw, x_experiment_raw = x[:reference_sample_size], x[reference_sample_size:]
    y_reference_raw, y_experiment_raw = y[:reference_sample_size], y[reference_sample_size:]

    # Mask NaNs independently in each block.
    reference_mask = ~(np.isnan(x_reference_raw) | np.isnan(y_reference_raw))
    experiment_mask = ~(np.isnan(x_experiment_raw) | np.isnan(y_experiment_raw))
    x_reference, y_reference = x_reference_raw[reference_mask], y_reference_raw[reference_mask]
    x_experiment, y_experiment = x_experiment_raw[experiment_mask], y_experiment_raw[experiment_mask]

    # Safety guard: if any block is empty, contribution is undefined.
    if x_reference.size == 0 or x_experiment.size == 0:
        return np.nan, np.nan, np.nan

    x_all = np.concatenate([x_reference, x_experiment])
    y_all = np.concatenate([y_reference, y_experiment])

    # means
    x_mean, y_mean = x_all.mean(), y_all.mean()

    # mean distance for reference and experiment samples for gene1
    x_reference_centered = x_reference - x_mean
    x_experiment_centered = x_experiment - x_mean
    # mean distance for reference and experiment samples for gene2
    y_reference_centered = y_reference - y_mean
    y_experiment_centered = y_experiment - y_mean

    # mean distance for vector gene1 and gene2
    xm, ym = x_all - x_mean, y_all - y_mean

    # np.linalg.norm is the L2-Norm: calculates Euclidean distance for given vector
    # we will use this value as a denominator
    normxm = np.linalg.norm(xm)
    normym = np.linalg.norm(ym)

    # np.dot is matrix multiplication
    # multiplicates A and B vectors and divedes to L2=norm value (A*B)
    denom = normxm * normym

    # Safety guard: avoid division by zero/invalid denominator.
    if not np.isfinite(denom) or np.isclose(denom, 0.0):
        return np.nan, np.nan, np.nan

    reference_contr_val = np.dot(x_reference_centered, y_reference_centered)
    experiment_contr_val = np.dot(x_experiment_centered, y_experiment_centered)
    r = np.dot(xm, ym) / denom
    reference_contr = reference_contr_val / denom
    experiment_contr = experiment_contr_val / denom

    return r, reference_contr, experiment_contr


def get_contributions(df, reference_sample_size):
    # I don't remember why I calculated std here.
    # std = df.T.std(ddof=1)
    # std_matrix = pd.DataFrame(np.outer(std, std), columns=df.index, index=df.index)

    # Mean expression across all samples
    global_mean = df.T.mean()
    # is this my addition or Max asked for it?
    df = df.apply(lambda x: x - global_mean)

    # Split into reference and experiment groups
    reference_mean = df.iloc[:, :reference_sample_size]
    experiment_mean = df.iloc[:, reference_sample_size:]

    # Dot products for each group (co-deviation matrix)
    reference_contr = pd.DataFrame(np.dot(reference_mean, reference_mean.T), columns=df.index, index=df.index)
    experiment_contr = pd.DataFrame(np.dot(experiment_mean, experiment_mean.T), columns=df.index, index=df.index)

    # Normalize by Euclidean norm of mean-centered expression vectors
    norms = df.T.apply(np.linalg.norm)
    denom = pd.DataFrame(np.outer(norms, norms), columns=df.index, index=df.index)

    # Final normalized contribution matrices
    reference_contr /= denom
    experiment_contr /= denom

    # Remove diagonal (self-contribution)
    np.fill_diagonal(reference_contr.values, np.nan)
    np.fill_diagonal(experiment_contr.values, np.nan)

    return reference_contr, experiment_contr


def get_differential_correlation(merged_df, reference_sample_size):
    """
    Differential correlation: rM - rR.

    Returns:
      diff_corr: matrix of correlation changes (rM - rR); diagonal = NaN
      direction: directional contribution matrix with values {-1, 0, 1} off-diagonal
    """
    reference_df = merged_df.iloc[:, :reference_sample_size]

    rM = perform_corr(merged_df, "numpy")
    rR = perform_corr(reference_df, "numpy")

    diff_corr = rM - rR

    direction = pd.DataFrame(0, index=rM.index, columns=rM.columns, dtype=float)
    direction[(rM > 0) & (rR > 0) & (rM > rR)] = 1
    direction[(rM < 0) & (rR < 0) & (rM < rR)] = -1
    np.fill_diagonal(direction.values, np.nan)

    return diff_corr, direction


def get_covariation_contributions(df, reference_sample_size):
    """
    Covariation-based contribution using global mean centering and
    scalar sample-size normalization per group.
    """
    n_reference = reference_sample_size
    n_experiment = df.shape[1] - n_reference

    global_mean = df.T.mean()
    mean_centered = df.apply(lambda x: x - global_mean)

    reference_mean = mean_centered.iloc[:, :n_reference]
    experiment_mean = mean_centered.iloc[:, n_reference:]

    reference_contr = pd.DataFrame(np.dot(reference_mean, reference_mean.T), columns=df.index, index=df.index)
    experiment_contr = pd.DataFrame(np.dot(experiment_mean, experiment_mean.T), columns=df.index, index=df.index)

    reference_contr /= n_reference
    experiment_contr /= n_experiment

    np.fill_diagonal(reference_contr.values, np.nan)
    np.fill_diagonal(experiment_contr.values, np.nan)

    return reference_contr, experiment_contr


def get_final_score(contribution_score, merged_df):
    """
    Final score = contribution_score * corr(M) as an element-wise product.
    """
    corr_M = perform_corr(merged_df, "numpy")
    return contribution_score * corr_M


__all__ = [
    "pairwise_contribution",
    "get_contributions",
    "get_differential_correlation",
    "get_covariation_contributions",
    "get_final_score",
]