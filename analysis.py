
import glob
import os
import random
from tqdm import tqdm 
import numpy as np
import pandas as pd
from sklearn import preprocessing
from statsmodels.stats.proportion import proportions_ztest
from pathlib import Path
from logging_config import log
import re
random.seed(13)
tqdm.pandas()




def perform_corr(df, corr_func="numpy"):
    if corr_func not in {"numpy", "pandas"}:
        raise ValueError("corr_func must be 'numpy' or 'pandas'")

    log.started(f"Performing correlation using '{corr_func}' method.")
    if corr_func == "numpy":
        corr_values = np.corrcoef(df.values)
        np.fill_diagonal(corr_values, np.nan)
        corr = pd.DataFrame(corr_values, index=df.index, columns=df.index)
        log.done("Correlation.")
        return corr
    else:
        corr = df.T.corr()
        np.fill_diagonal(corr.values, np.nan)
        return corr



## main function for contribution calculation
def get_contributions(df, sample_size_separation):
    
    # I don't remember why I calculated std here.

    #std = df.T.std(ddof=1)
    #std_matrix = pd.DataFrame(np.outer(std, std), columns=df.index, index=df.index)

    # Mean expression across all samples
    global_mean = df.T.mean()

    # is this my addition or Max asked for it?
    mean_centered = df.apply(lambda x: x - global_mean)

    # Split into big and small groups
    big_mean = mean_centered.iloc[:, :sample_size_separation]
    small_mean = mean_centered.iloc[:, sample_size_separation:]

    # Dot products for each group (co-deviation matrix)
    contr_big = pd.DataFrame(np.dot(big_mean, big_mean.T), columns=df.index, index=df.index)
    contr_small = pd.DataFrame(np.dot(small_mean, small_mean.T), columns=df.index, index=df.index)

    # Normalize by Euclidean norm of mean-centered expression vectors
    norms = mean_centered.T.apply(np.linalg.norm)
    denom = pd.DataFrame(np.outer(norms, norms), columns=df.index, index=df.index)

    # Final normalized contribution matrices
    contr_big /= denom
    contr_small /= denom

    # Remove diagonal (self-contribution)
    np.fill_diagonal(contr_big.values, np.nan)
    np.fill_diagonal(contr_small.values, np.nan)

    return contr_big, contr_small


def get_differential_correlation(merged_df, sample_size_separation):
    """
    Differential correlation: rM - rR.

    Returns:
      diff_corr: matrix of correlation changes (rM - rR); diagonal = NaN
      direction: directional contribution matrix with values {-1, 0, 1} off-diagonal
    """
    R = merged_df.iloc[:, :sample_size_separation]

    rM = perform_corr(merged_df)
    rR = perform_corr(R)

    diff_corr = rM - rR

    direction = pd.DataFrame(0, index=rM.index, columns=rM.columns, dtype=float)
    direction[(rM > 0) & (rR > 0) & (rM > rR)] = 1
    direction[(rM < 0) & (rR < 0) & (rM < rR)] = -1
    np.fill_diagonal(direction.values, np.nan)

    return diff_corr, direction


def get_covariation_contributions(df, sample_size_separation):
    """
    Covariation-based contribution using global mean centering and
    scalar sample-size normalization per group.
    """
    n_R = sample_size_separation
    n_T = df.shape[1] - n_R

    global_mean = df.T.mean()
    mean_centered = df.apply(lambda x: x - global_mean)

    big_mean = mean_centered.iloc[:, :n_R]
    small_mean = mean_centered.iloc[:, n_R:]

    contr_big = pd.DataFrame(np.dot(big_mean, big_mean.T), columns=df.index, index=df.index)
    contr_small = pd.DataFrame(np.dot(small_mean, small_mean.T), columns=df.index, index=df.index)

    contr_big /= n_R
    contr_small /= n_T

    np.fill_diagonal(contr_big.values, np.nan)
    np.fill_diagonal(contr_small.values, np.nan)

    return contr_big, contr_small


def get_final_score(contribution_score, merged_df):
    """
    Final score = contribution_score * corr(M) as an element-wise product.
    """
    corr_M = perform_corr(merged_df)
    return contribution_score * corr_M


# just for validation 
# we can check A and B gene to get their contribution
def pairwise_contribution(x, y, column):
    # x = gene1 vector
    # y = gene2 vector
    # column seperates dataset i.e: 7

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Split at the boundary before masking to preserve block identity.
    xA_raw, xB_raw = x[:column], x[column:]
    yA_raw, yB_raw = y[:column], y[column:]

    # Mask NaNs independently in each block.
    maskA = ~(np.isnan(xA_raw) | np.isnan(yA_raw))
    maskB = ~(np.isnan(xB_raw) | np.isnan(yB_raw))
    xA, yA = xA_raw[maskA], yA_raw[maskA]
    xB, yB = xB_raw[maskB], yB_raw[maskB]

    # Safety guard: if any block is empty, contribution is undefined.
    if xA.size == 0 or xB.size == 0:
        return np.nan, np.nan, np.nan

    x_all = np.concatenate([xA, xB])
    y_all = np.concatenate([yA, yB])

    # means
    x_mean, y_mean = x_all.mean(), y_all.mean()

    # mean distance  each sample group A and B for gene1
    xAm, xBm = xA - x_mean, xB - x_mean
    # mean distance  each sample group A and B for gene2
    yAm, yBm = yA - y_mean, yB - y_mean

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

    big_contr_val = np.dot(xAm, yAm)
    small_contr_val = np.dot(xBm, yBm)
    r = np.dot(xm, ym) / denom
    big_contr = big_contr_val / denom  # big contr.
    small_contr = small_contr_val / denom  # small contr.

    return r, big_contr, small_contr

        

## DepMap tissue type annotation

def slug(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^\w\s-]", "", text)  # remove non-word characters
    text = re.sub(r"[\s_]+", "-", text)  # replace spaces/underscores with hyphen
    return text.strip("-")


def create_annotation(filename, min_sample=0, tissue_type="OncotreeLineage"):
    # Load DepMap model data
    annotation = pd.read_csv(filename, index_col=0)

    # Clean tissue labels if using OncotreeLineage
    if tissue_type == "OncotreeLineage":
        conditions = [
            (annotation["OncotreePrimaryDisease"] == "Melanoma", "Melanoma"),
            (annotation["OncotreeLineage"] == "Skin", "Skin Other"),
            (annotation["OncotreePrimaryDisease"] == "Non-Small Cell Lung Cancer", "NSCLC"),
            (annotation["OncotreePrimaryDisease"] == "Lung Neuroendocrine Tumor", "SCLC"),
            (
                (annotation["OncotreePrimaryDisease"] == "Non-Cancerous") & 
                (annotation["OncotreeLineage"] == "Lung"),
                "Lung Non-Cancerous"
            ),
            (annotation["OncotreePrimaryDisease"] == "Neuroblastoma", "Neuroblastoma"),
        ]
        for condition, value in conditions:
            annotation.loc[condition, "OncotreeLineage"] = value

    # Group by tissue type and count samples
    annotation = (
        annotation[[tissue_type]]
        .reset_index()
        .groupby(tissue_type, as_index=False)
        .agg({"ModelID": ["count", list]})
    )
    annotation.columns = ["tissue", "depmap_nsample", "sample_list"]
    annotation = annotation.query(f"depmap_nsample >= {min_sample}").sort_values(by="tissue")
    annotation['slug'] =  [slug(t) for t in annotation["tissue"]]
    all_samples = annotation["sample_list"].explode().to_list()
    return annotation, all_samples


def read_annotation(filename="data/model_preprocessed.csv"):
    model = pd.read_csv(filename)
    samples = model.sample_list.explode().to_list()
    return model, samples





