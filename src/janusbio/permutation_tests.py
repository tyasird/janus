import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def _validate_adaptive_topk_inputs(
    merged_df,
    reference_sample_size,
    n_experiment,
    top_k,
    n_perm,
):
    if not isinstance(merged_df, pd.DataFrame):
        raise TypeError("merged_df must be a pandas DataFrame.")

    if merged_df.shape[0] < 2:
        raise ValueError("merged_df must contain at least 2 genes (rows).")

    n_samples = merged_df.shape[1]
    if n_samples < 2:
        raise ValueError("merged_df must contain at least 2 samples (columns).")

    if not isinstance(reference_sample_size, int):
        raise TypeError("reference_sample_size must be an integer.")

    if reference_sample_size <= 0 or reference_sample_size >= n_samples:
        raise ValueError(
            "reference_sample_size must be between 1 and total_samples - 1."
        )

    if n_experiment is None:
        n_experiment = n_samples - reference_sample_size

    if not isinstance(n_experiment, int):
        raise TypeError("n_experiment must be an integer or None.")

    if n_experiment <= 0 or n_experiment >= n_samples:
        raise ValueError("n_experiment must be between 1 and total_samples - 1.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if not isinstance(n_perm, int) or n_perm <= 0:
        raise ValueError("n_perm must be a positive integer.")

    return n_experiment, top_k, n_perm


def _precompute_adaptive_topk_arrays(merged_df):
    """
    Precompute arrays shared by all permutations.

    Uses the same core idea as get_contributions():
    - global mean center each gene across all samples
    - denominator comes from full merged matrix norms
    """
    values = merged_df.to_numpy(dtype=float, copy=True)

    values -= values.mean(axis=1, keepdims=True)

    norms = np.linalg.norm(values, axis=1)

    tri_rows, tri_cols = np.triu_indices(values.shape[0], k=1)

    tri_denom = norms[tri_rows] * norms[tri_cols]
    valid_tri = np.isfinite(tri_denom) & (~np.isclose(tri_denom, 0.0))

    return values, tri_rows, tri_cols, tri_denom, valid_tri


def _extract_topk_positive_from_edge_values(
    edge_values,
    tri_rows,
    tri_cols,
    k_limit,
    return_pairs=False,
):
    """
    From a vector of upper-triangle edge values:
    - keep only positive finite values
    - select largest top-k using argpartition
    - sort only selected top-k descending
    """
    positive_idx = np.flatnonzero(np.isfinite(edge_values) & (edge_values > 0))

    if positive_idx.size == 0:
        if return_pairs:
            return (
                np.empty(0, dtype=float),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
            )
        return np.empty(0, dtype=float), None, None

    positive_values = edge_values[positive_idx]
    k_eff = min(k_limit, positive_values.size)

    if k_eff == positive_values.size:
        selected_local_idx = np.arange(positive_values.size)
    else:
        selected_local_idx = np.argpartition(positive_values, -k_eff)[-k_eff:]

    selected_values = positive_values[selected_local_idx]
    order = np.argsort(selected_values)[::-1]
    selected_tri_idx = positive_idx[selected_local_idx[order]]

    topk_values = edge_values[selected_tri_idx]

    if return_pairs:
        row_idx = tri_rows[selected_tri_idx]
        col_idx = tri_cols[selected_tri_idx]
        return topk_values, row_idx, col_idx

    return topk_values, None, None


def _extract_topk_positive_from_subset(
    centered,
    tri_rows,
    tri_cols,
    tri_denom,
    valid_tri,
    sample_indices,
    k_limit,
    return_pairs=False,
):
    """
    Compute experiment-subset contribution values only once for the selected subset,
    then extract top-k positive upper-triangle values.
    """
    subset = centered[:, sample_indices]
    cross = subset @ subset.T

    raw_edge_values = cross[tri_rows, tri_cols]

    edge_values = np.full(raw_edge_values.shape, np.nan, dtype=float)
    np.divide(raw_edge_values, tri_denom, out=edge_values, where=valid_tri)

    return _extract_topk_positive_from_edge_values(
        edge_values=edge_values,
        tri_rows=tri_rows,
        tri_cols=tri_cols,
        k_limit=k_limit,
        return_pairs=return_pairs,
    )


def _run_adaptive_topk_permutation_chunk(
    centered,
    tri_rows,
    tri_cols,
    tri_denom,
    valid_tri,
    n_samples,
    n_experiment,
    effective_k,
    top_k,
    obs_topk,
    obs_mean,
    obs_sum,
    obs_min,
    n_perm,
    random_state,
    create_table_with_perm_values=False,
    create_table_with_selected_samples=False,
    sample_names=None,
    perm_id_start=1,
):
    rank_counts = np.zeros(effective_k, dtype=np.int64)
    rank_valid = np.zeros(effective_k, dtype=np.int64)

    mean_count = 0
    sum_count = 0
    min_count = 0

    mean_valid = 0
    sum_valid = 0
    min_valid = 0

    permuted_value_records = []
    selected_sample_records = []
    rng = np.random.default_rng(random_state)

    for local_perm_id in range(n_perm):
        perm_experiment_idx = rng.choice(n_samples, size=n_experiment, replace=False)

        if create_table_with_selected_samples:
            current_sample_names = sample_names if sample_names is not None else np.arange(n_samples)
            selected_names = [current_sample_names[idx] for idx in perm_experiment_idx]
            selected_sample_records.append(
                {
                    "perm_id": perm_id_start + local_perm_id,
                    "n_selected": int(len(selected_names)),
                    "selected_samples": selected_names,
                }
            )

        perm_topk, _, _ = _extract_topk_positive_from_subset(
            centered=centered,
            tri_rows=tri_rows,
            tri_cols=tri_cols,
            tri_denom=tri_denom,
            valid_tri=valid_tri,
            sample_indices=perm_experiment_idx,
            k_limit=top_k,
            return_pairs=False,
        )

        if create_table_with_perm_values:
            permuted_value_records.append(
                {
                    "perm_id": perm_id_start + local_perm_id,
                    "n_values": int(perm_topk.size),
                    "values": perm_topk.tolist(),
                }
            )

        n_compare = min(effective_k, perm_topk.size)
        if n_compare > 0:
            rank_valid[:n_compare] += 1
            rank_counts[:n_compare] += (
                perm_topk[:n_compare] >= obs_topk[:n_compare]
            ).astype(np.int64)

        if perm_topk.size >= effective_k:
            aligned = perm_topk[:effective_k]
            perm_mean = float(np.mean(aligned))
            perm_sum = float(np.sum(aligned))
            perm_min = float(aligned[-1])

            mean_valid += 1
            sum_valid += 1
            min_valid += 1

            mean_count += int(perm_mean >= obs_mean)
            sum_count += int(perm_sum >= obs_sum)
            min_count += int(perm_min >= obs_min)

    return {
        "rank_counts": rank_counts,
        "rank_valid": rank_valid,
        "summary_counts": np.array([mean_count, sum_count, min_count], dtype=np.int64),
        "summary_valid": np.array([mean_valid, sum_valid, min_valid], dtype=np.int64),
        "permuted_value_records": permuted_value_records,
        "selected_sample_records": selected_sample_records,
    }


def adaptive_topk_rank_permutation_test(
    merged_df,
    reference_sample_size,
    n_experiment=None,
    top_k=10000,
    n_perm=100,
    random_state=13,
    create_table_with_perm_values=False,
    create_table_with_selected_samples=False,
    sanity_check=False,
):
    """
    Adaptive top-k rank-order permutation test on the experiment contribution matrix.

    This is rank comparison only, not edge-identity comparison.

    Observed step
    -------------
    - Use the true observed experiment subset (the columns after reference_sample_size)
    - Compute the experiment contribution structure
    - Extract the top-k positive edge values
    - Sort them descending

    Permutation step
    ----------------
    - Randomly choose n_experiment samples from all samples
    - Recompute the experiment contribution structure
    - Re-select top-k positive edge values
    - Sort them descending
    - Compare rank 1 with rank 1, rank 2 with rank 2, etc.

    Empirical p-value
    -----------------
    For each rank:
        p = (count_ge_observed + 1) / (n_valid_perm + 1)

    Summary statistics use the same empirical p-value formula, but only over
    permutations that reach the same rank depth as the observed top-k.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        Gene x sample merged matrix.
    reference_sample_size : int
        Number of reference samples at the start of the merged reference + experiment matrix.
    n_experiment : int or None
        Size of the experiment subset in each permutation. If None, inferred from observed data.
    top_k : int
        Number of strongest positive edges to keep.
    n_perm : int
        Number of random permutations.
    random_state : int
        RNG seed.
    create_table_with_perm_values : bool
        If True, also return a compact table of permuted top-k vectors.
    create_table_with_selected_samples : bool
        If True, also return a table with the selected sample names for each permutation.
    sanity_check : bool
        If True, compare helper result to direct contribution computation once.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys:
        - rank_pvalues
        - summary_pvalues
        - permuted_values (optional)
    """
    n_experiment, top_k, n_perm = _validate_adaptive_topk_inputs(
        merged_df=merged_df,
        reference_sample_size=reference_sample_size,
        n_experiment=n_experiment,
        top_k=top_k,
        n_perm=n_perm,
    )

    centered, tri_rows, tri_cols, tri_denom, valid_tri = _precompute_adaptive_topk_arrays(
        merged_df
    )

    n_samples = centered.shape[1]
    observed_experiment_idx = np.arange(reference_sample_size, n_samples)

    obs_topk, _, _ = _extract_topk_positive_from_subset(
        centered=centered,
        tri_rows=tri_rows,
        tri_cols=tri_cols,
        tri_denom=tri_denom,
        valid_tri=valid_tri,
        sample_indices=observed_experiment_idx,
        k_limit=top_k,
        return_pairs=False,
    )

    if obs_topk.size == 0:
        raise ValueError(
            "Observed experiment subset has no positive edges. Rank-based testing is undefined."
        )

    effective_k = obs_topk.size

    if sanity_check:
        values = merged_df.to_numpy(dtype=float, copy=True)
        values -= values.mean(axis=1, keepdims=True)
        experiment_values = values[:, reference_sample_size:]

        direct_cross = experiment_values @ experiment_values.T
        norms = np.linalg.norm(values, axis=1)
        direct_denom = np.outer(norms, norms)

        direct_experiment = np.full(direct_cross.shape, np.nan, dtype=float)
        valid_direct = np.isfinite(direct_denom) & (~np.isclose(direct_denom, 0.0))
        np.divide(direct_cross, direct_denom, out=direct_experiment, where=valid_direct)

        direct_edge_values = direct_experiment[tri_rows, tri_cols]
        direct_topk, _, _ = _extract_topk_positive_from_edge_values(
            edge_values=direct_edge_values,
            tri_rows=tri_rows,
            tri_cols=tri_cols,
            k_limit=top_k,
            return_pairs=False,
        )

        if obs_topk.size != direct_topk.size or not np.allclose(
            obs_topk, direct_topk, atol=1e-10, rtol=1e-10
        ):
            raise ValueError(
                "Sanity check failed: helper-based observed top-k does not match "
                "direct observed top-k."
            )

    obs_mean = float(np.mean(obs_topk))
    obs_sum = float(np.sum(obs_topk))
    obs_min = float(obs_topk[-1])

    rank_counts = np.zeros(effective_k, dtype=np.int64)
    rank_valid = np.zeros(effective_k, dtype=np.int64)

    mean_count = 0
    sum_count = 0
    min_count = 0

    mean_valid = 0
    sum_valid = 0
    min_valid = 0
    permuted_value_records = []
    selected_sample_records = []
    sample_names = merged_df.columns.to_numpy()

    rng = np.random.default_rng(random_state)

    for perm_id in range(1, n_perm + 1):
        perm_experiment_idx = rng.choice(n_samples, size=n_experiment, replace=False)

        if create_table_with_selected_samples:
            selected_names = [sample_names[idx] for idx in perm_experiment_idx]
            selected_sample_records.append(
                {
                    "perm_id": perm_id,
                    "n_selected": int(len(selected_names)),
                    "selected_samples": selected_names,
                }
            )

        perm_topk, _, _ = _extract_topk_positive_from_subset(
            centered=centered,
            tri_rows=tri_rows,
            tri_cols=tri_cols,
            tri_denom=tri_denom,
            valid_tri=valid_tri,
            sample_indices=perm_experiment_idx,
            k_limit=top_k,
            return_pairs=False,
        )

        if create_table_with_perm_values:
            permuted_value_records.append(
                {
                    "perm_id": perm_id,
                    "n_values": int(perm_topk.size),
                    "values": perm_topk.tolist(),
                }
            )

        n_compare = min(effective_k, perm_topk.size)
        if n_compare > 0:
            rank_valid[:n_compare] += 1
            rank_counts[:n_compare] += (
                perm_topk[:n_compare] >= obs_topk[:n_compare]
            ).astype(np.int64)

        if perm_topk.size >= effective_k:
            aligned = perm_topk[:effective_k]
            perm_mean = float(np.mean(aligned))
            perm_sum = float(np.sum(aligned))
            perm_min = float(aligned[-1])

            mean_valid += 1
            sum_valid += 1
            min_valid += 1

            mean_count += int(perm_mean >= obs_mean)
            sum_count += int(perm_sum >= obs_sum)
            min_count += int(perm_min >= obs_min)

        else:
            perm_mean = np.nan
            perm_sum = np.nan
            perm_min = np.nan

    rank_p = np.full(effective_k, np.nan, dtype=float)
    valid_mask = rank_valid > 0
    rank_p[valid_mask] = (rank_counts[valid_mask] + 1) / (rank_valid[valid_mask] + 1)

    rank_pvalues_df = pd.DataFrame(
        {
            "rank": np.arange(1, effective_k + 1, dtype=int),
            "observed_value": obs_topk,
            "n_perm_ge_observed": rank_counts,
            "p_empirical": rank_p,
        }
    )

    summary_counts = np.array([mean_count, sum_count, min_count], dtype=np.int64)
    summary_valid = np.array([mean_valid, sum_valid, min_valid], dtype=np.int64)

    summary_p = np.full(3, np.nan, dtype=float)
    valid_summary_mask = summary_valid > 0
    summary_p[valid_summary_mask] = (
        (summary_counts[valid_summary_mask] + 1)
        / (summary_valid[valid_summary_mask] + 1)
    )

    summary_pvalues_df = pd.DataFrame(
        {
            "statistic": ["mean", "sum", "min"],
            "observed": [obs_mean, obs_sum, obs_min],
            "n_perm_ge_observed": summary_counts,
            "n_perm_valid": summary_valid,
            "n_perm_requested": [n_perm, n_perm, n_perm],
            "p_empirical": summary_p,
            "top_k_requested": [top_k, top_k, top_k],
            "effective_k": [effective_k, effective_k, effective_k],
            "alternative": ["greater", "greater", "greater"],
            "test_type": [
                "adaptive_topk_rank_permutation",
                "adaptive_topk_rank_permutation",
                "adaptive_topk_rank_permutation",
            ],
        }
    )

    results = {
        "rank_pvalues": rank_pvalues_df,
        "summary_pvalues": summary_pvalues_df,
    }

    if create_table_with_perm_values:
        results["permuted_values"] = pd.DataFrame(permuted_value_records)

    if create_table_with_selected_samples:
        results["selected_samples"] = pd.DataFrame(selected_sample_records)

    return results


def adaptive_topk_rank_permutation_test_parallel(
    merged_df,
    reference_sample_size,
    n_experiment=None,
    top_k=10000,
    n_perm=100,
    random_state=13,
    create_table_with_perm_values=False,
    create_table_with_selected_samples=False,
    sanity_check=False,
    n_jobs=None,
    permutations_per_job=None,
):
    """
    Parallel wrapper for adaptive_topk_rank_permutation_test.

    This function leaves the original implementation untouched and parallelizes
    only the permutation loop by splitting permutations into independent chunks
    whose aggregate counts are merged at the end.
    """
    n_experiment, top_k, n_perm = _validate_adaptive_topk_inputs(
        merged_df=merged_df,
        reference_sample_size=reference_sample_size,
        n_experiment=n_experiment,
        top_k=top_k,
        n_perm=n_perm,
    )

    centered, tri_rows, tri_cols, tri_denom, valid_tri = _precompute_adaptive_topk_arrays(
        merged_df
    )

    n_samples = centered.shape[1]
    observed_experiment_idx = np.arange(reference_sample_size, n_samples)

    obs_topk, _, _ = _extract_topk_positive_from_subset(
        centered=centered,
        tri_rows=tri_rows,
        tri_cols=tri_cols,
        tri_denom=tri_denom,
        valid_tri=valid_tri,
        sample_indices=observed_experiment_idx,
        k_limit=top_k,
        return_pairs=False,
    )

    if obs_topk.size == 0:
        raise ValueError(
            "Observed experiment subset has no positive edges. Rank-based testing is undefined."
        )

    effective_k = obs_topk.size

    if sanity_check:
        reference_result = adaptive_topk_rank_permutation_test(
            merged_df=merged_df,
            reference_sample_size=reference_sample_size,
            n_experiment=n_experiment,
            top_k=top_k,
            n_perm=1,
            random_state=random_state,
            create_table_with_perm_values=False,
            sanity_check=True,
        )
        if not np.allclose(
            reference_result["rank_pvalues"]["observed_value"].to_numpy(),
            obs_topk,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError(
                "Sanity check failed: parallel observed top-k does not match reference implementation."
            )

    obs_mean = float(np.mean(obs_topk))
    obs_sum = float(np.sum(obs_topk))
    obs_min = float(obs_topk[-1])

    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() or 1)
    n_jobs = max(1, min(int(n_jobs), n_perm))

    if permutations_per_job is None:
        base = n_perm // n_jobs
        remainder = n_perm % n_jobs
        chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_jobs)]
        chunk_sizes = [size for size in chunk_sizes if size > 0]
    else:
        if not isinstance(permutations_per_job, int) or permutations_per_job <= 0:
            raise ValueError("permutations_per_job must be a positive integer or None.")
        chunk_sizes = []
        remaining = n_perm
        while remaining > 0:
            current = min(permutations_per_job, remaining)
            chunk_sizes.append(current)
            remaining -= current

    seed_sequence = np.random.SeedSequence(random_state)
    child_seeds = seed_sequence.spawn(len(chunk_sizes))

    rank_counts = np.zeros(effective_k, dtype=np.int64)
    rank_valid = np.zeros(effective_k, dtype=np.int64)
    summary_counts = np.zeros(3, dtype=np.int64)
    summary_valid = np.zeros(3, dtype=np.int64)
    permuted_value_records = []
    selected_sample_records = []
    sample_names = merged_df.columns.to_numpy()

    futures = []
    perm_id_start = 1
    with ThreadPoolExecutor(max_workers=min(n_jobs, len(chunk_sizes))) as executor:
        for chunk_size, child_seed in zip(chunk_sizes, child_seeds):
            worker_seed = int(child_seed.generate_state(1, dtype=np.uint64)[0])
            futures.append(
                executor.submit(
                    _run_adaptive_topk_permutation_chunk,
                    centered,
                    tri_rows,
                    tri_cols,
                    tri_denom,
                    valid_tri,
                    n_samples,
                    n_experiment,
                    effective_k,
                    top_k,
                    obs_topk,
                    obs_mean,
                    obs_sum,
                    obs_min,
                    chunk_size,
                    worker_seed,
                    create_table_with_perm_values,
                    create_table_with_selected_samples,
                    sample_names,
                    perm_id_start,
                )
            )
            perm_id_start += chunk_size

        for future in futures:
            chunk_result = future.result()
            rank_counts += chunk_result["rank_counts"]
            rank_valid += chunk_result["rank_valid"]
            summary_counts += chunk_result["summary_counts"]
            summary_valid += chunk_result["summary_valid"]
            if create_table_with_perm_values:
                permuted_value_records.extend(chunk_result["permuted_value_records"])
            if create_table_with_selected_samples:
                selected_sample_records.extend(chunk_result["selected_sample_records"])

    rank_p = np.full(effective_k, np.nan, dtype=float)
    valid_mask = rank_valid > 0
    rank_p[valid_mask] = (rank_counts[valid_mask] + 1) / (rank_valid[valid_mask] + 1)

    rank_pvalues_df = pd.DataFrame(
        {
            "rank": np.arange(1, effective_k + 1, dtype=int),
            "observed_value": obs_topk,
            "n_perm_ge_observed": rank_counts,
            "p_empirical": rank_p,
        }
    )

    summary_p = np.full(3, np.nan, dtype=float)
    valid_summary_mask = summary_valid > 0
    summary_p[valid_summary_mask] = (
        (summary_counts[valid_summary_mask] + 1)
        / (summary_valid[valid_summary_mask] + 1)
    )

    summary_pvalues_df = pd.DataFrame(
        {
            "statistic": ["mean", "sum", "min"],
            "observed": [obs_mean, obs_sum, obs_min],
            "n_perm_ge_observed": summary_counts,
            "n_perm_valid": summary_valid,
            "n_perm_requested": [n_perm, n_perm, n_perm],
            "p_empirical": summary_p,
            "top_k_requested": [top_k, top_k, top_k],
            "effective_k": [effective_k, effective_k, effective_k],
            "alternative": ["greater", "greater", "greater"],
            "test_type": [
                "adaptive_topk_rank_permutation_parallel",
                "adaptive_topk_rank_permutation_parallel",
                "adaptive_topk_rank_permutation_parallel",
            ],
        }
    )

    results = {
        "rank_pvalues": rank_pvalues_df,
        "summary_pvalues": summary_pvalues_df,
    }

    if create_table_with_perm_values:
        permuted_values_df = pd.DataFrame(permuted_value_records)
        if not permuted_values_df.empty:
            permuted_values_df = permuted_values_df.sort_values("perm_id").reset_index(drop=True)
        results["permuted_values"] = permuted_values_df

    if create_table_with_selected_samples:
        selected_samples_df = pd.DataFrame(selected_sample_records)
        if not selected_samples_df.empty:
            selected_samples_df = selected_samples_df.sort_values("perm_id").reset_index(drop=True)
        results["selected_samples"] = selected_samples_df

    return results


adaptive_topk_rank_permutation_test2 = adaptive_topk_rank_permutation_test


__all__ = [
    "adaptive_topk_rank_permutation_test",
    "adaptive_topk_rank_permutation_test_parallel",
    "adaptive_topk_rank_permutation_test2",
    "_precompute_adaptive_topk_arrays",
    "_extract_topk_positive_from_subset",
]
