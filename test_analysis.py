import os
import tempfile

import numpy as np
import pandas as pd

import analysis
import preprocessing as pp
import utils


def make_clean_data(n_genes=5, n_ref=10, n_target=3, seed=42):
    """Create small, NaN-free synthetic merged DataFrame."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_genes, n_ref + n_target))
    genes = [f"GENE{i}" for i in range(n_genes)]
    samples = [f"S{i}" for i in range(n_ref + n_target)]
    return pd.DataFrame(data, index=genes, columns=samples), n_ref


def test_pairwise_matches_matrix_on_clean_data():
    """pairwise_contribution() should match get_contributions() when no NaNs."""
    df, sep = make_clean_data()
    big, small = analysis.get_contributions(df, sep)

    for g1 in df.index:
        for g2 in df.index:
            if g1 == g2:
                continue
            r, bc, sc = analysis.pairwise_contribution(
                df.loc[g1].values, df.loc[g2].values, sep
            )
            assert abs(bc - big.loc[g1, g2]) < 1e-10, f"big mismatch at ({g1},{g2})"
            assert abs(sc - small.loc[g1, g2]) < 1e-10, f"small mismatch at ({g1},{g2})"
            assert np.isfinite(r), f"r should be finite at ({g1},{g2})"
    print("PASS: pairwise matches matrix on NaN-free data")


def test_pairwise_safety_guards():
    """pairwise_contribution() returns NaN triple on empty block or zero denominator."""
    # Empty first block after NaN masking
    x = np.array([np.nan, np.nan, 1.0, 2.0])
    y = np.array([np.nan, np.nan, 3.0, 4.0])
    r, bc, sc = analysis.pairwise_contribution(x, y, column=2)
    assert np.isnan(r) and np.isnan(bc) and np.isnan(sc)

    # Zero denominator from zero-variance vectors
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([2.0, 2.0, 2.0, 2.0])
    r, bc, sc = analysis.pairwise_contribution(x, y, column=2)
    assert np.isnan(r) and np.isnan(bc) and np.isnan(sc)

    print("PASS: pairwise safety guards work")


def test_return_shapes():
    """All functions return (n_genes x n_genes) matrices."""
    df, sep = make_clean_data()
    n = df.shape[0]

    big, small = analysis.get_contributions(df, sep)
    assert big.shape == small.shape == (n, n)

    diff, direction = analysis.get_differential_correlation(df, sep)
    assert diff.shape == direction.shape == (n, n)

    big_cov, small_cov = analysis.get_covariation_contributions(df, sep)
    assert big_cov.shape == small_cov.shape == (n, n)

    final = analysis.get_final_score(small, df)
    assert final.shape == (n, n)
    print("PASS: all shapes correct")


def test_direction_values():
    """direction matrix must only contain -1, 0, 1 off-diagonal and NaN diagonal."""
    df, sep = make_clean_data()
    _, direction = analysis.get_differential_correlation(df, sep)

    off_diag = direction.values[~np.eye(len(direction), dtype=bool)]
    assert set(off_diag).issubset({-1.0, 0.0, 1.0})
    assert all(np.isnan(np.diag(direction.values)))
    print("PASS: direction matrix values correct")


def test_return_order_consistency():
    """All contribution functions return (big, small) in the same order."""
    df, sep = make_clean_data()
    big1, small1 = analysis.get_contributions(df, sep)
    big2, small2 = analysis.get_covariation_contributions(df, sep)

    assert all(np.isnan(np.diag(big1.values)))
    assert all(np.isnan(np.diag(small1.values)))
    assert all(np.isnan(np.diag(big2.values)))
    assert all(np.isnan(np.diag(small2.values)))
    print("PASS: return order consistent, diagonals are NaN")


def test_drop_na_preprocessing_behavior():
    """Verify drop_na controls NaN row removal in load_datasets() without polluting project state."""
    df = pd.DataFrame(
        {
            "S1": [1.0, 2.0, np.nan],
            "S2": [3.0, 4.0, np.nan],
            "S3": [5.0, 6.0, 7.0],
        },
        index=["GENE0", "GENE1", "GENE_NAN"],
    )

    inputs = {"TestDataset": {"path": df, "sort": "high"}}

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            os.chdir(temp_dir)

            # drop_na=True -> NaN gene row must be removed
            utils.initialize({"preprocessing": {"drop_na": True}})
            data_true, _ = pp.load_datasets(inputs)
            assert "GENE_NAN" not in data_true["TestDataset"].index, (
                "NaN row should be dropped when drop_na=True"
            )

            # drop_na=False -> NaN gene row must survive
            utils.initialize({"preprocessing": {"drop_na": False}})
            data_false, _ = pp.load_datasets(inputs)
            assert "GENE_NAN" in data_false["TestDataset"].index, (
                "NaN row should be kept when drop_na=False"
            )
        finally:
            os.chdir(original_cwd)

    print("PASS: drop_na controls NaN row removal and test state is isolated")


if __name__ == "__main__":
    test_pairwise_matches_matrix_on_clean_data()
    test_pairwise_safety_guards()
    test_return_shapes()
    test_direction_values()
    test_return_order_consistency()
    test_drop_na_preprocessing_behavior()
    print("\nAll tests passed.")
