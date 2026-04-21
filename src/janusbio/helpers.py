import numpy as np
import pandas as pd
from numba import njit, prange

from .logging_config import log


def perform_corr(df, corr_func):
	if corr_func not in {"numpy", "numpy_without_mask", "pandas", "numba"}:
		raise ValueError("corr_func must be 'numpy' or 'pandas'")

	log.started(f"Performing correlation using '{corr_func}' method.")

	x_axis, y_axis = df.shape
	log.info(f"Data shape: {x_axis} features, {y_axis} samples")

	if corr_func == "numpy":
		m = np.ma.masked_invalid(df.values)
		corr = np.ma.corrcoef(m)
		arr = corr.filled(np.nan)
		df_corr = pd.DataFrame(arr, index=df.index, columns=df.index)
		np.fill_diagonal(df_corr.values, np.nan)
		if df_corr.shape != (x_axis, x_axis):
			raise ValueError(
				f"Correlation matrix shape mismatch: expected ({x_axis}, {x_axis}), got {df_corr.shape}"
			)
		log.done("Correlation.")
		return df_corr

	if corr_func == "numpy_without_mask":
		corr = np.corrcoef(df.values)
		df_corr = pd.DataFrame(corr, index=df.index, columns=df.index)
		np.fill_diagonal(df_corr.values, np.nan)
		if df_corr.shape != (x_axis, x_axis):
			raise ValueError(
				f"Correlation matrix shape mismatch: expected ({x_axis}, {x_axis}), got {df_corr.shape}"
			)
		log.done("Correlation.")
		return df_corr

	if corr_func == "numba":
		corr = fast_corr(df)
		np.fill_diagonal(corr.values, np.nan)
		if corr.shape != (x_axis, x_axis):
			raise ValueError(
				f"Correlation matrix shape mismatch: expected ({x_axis}, {x_axis}), got {corr.shape}"
			)
		log.done("Correlation using Numba.")
		return corr

	corr = df.T.corr()
	np.fill_diagonal(corr.values, np.nan)
	if corr.shape != (x_axis, x_axis):
		raise ValueError(
			f"Correlation matrix shape mismatch: expected ({x_axis}, {x_axis}), got {corr.shape}"
		)
	return corr


def fast_corr(df):
	@njit(parallel=True)
	def compute_corr(data):
		m, n = data.shape
		corr = np.full((n, n), np.nan, dtype=np.float64)

		for i in prange(n):
			for j in range(i + 1, n):
				sum_x = 0.0
				sum_y = 0.0
				sum_xx = 0.0
				sum_yy = 0.0
				sum_xy = 0.0
				count = 0
				for k in range(m):
					x = data[k, i]
					y = data[k, j]
					if not np.isnan(x) and not np.isnan(y):
						sum_x += x
						sum_y += y
						sum_xx += x * x
						sum_yy += y * y
						sum_xy += x * y
						count += 1
				if count >= 2:
					var_x = (sum_xx - (sum_x ** 2) / count) / (count - 1)
					var_y = (sum_yy - (sum_y ** 2) / count) / (count - 1)
					cov = (sum_xy - (sum_x * sum_y) / count) / (count - 1)
					denom = np.sqrt(var_x * var_y)
					if denom > 0:
						r = cov / denom
					else:
						r = np.nan
				else:
					r = np.nan
				corr[i, j] = r
				corr[j, i] = r

		for i in prange(n):
			sum_x = 0.0
			sum_xx = 0.0
			count = 0
			for k in range(m):
				x = data[k, i]
				if not np.isnan(x):
					sum_x += x
					sum_xx += x * x
					count += 1
			if count >= 2:
				var_x = (sum_xx - (sum_x ** 2) / count) / (count - 1)
				if var_x > 0:
					corr[i, i] = 1.0
				else:
					corr[i, i] = np.nan
			else:
				corr[i, i] = np.nan
		return corr

	df_numeric = df.select_dtypes(include=np.number)
	data = df_numeric.to_numpy().T
	corr_matrix = compute_corr(data)
	corr_df = pd.DataFrame(corr_matrix, index=df_numeric.index, columns=df_numeric.index)
	return corr_df


def is_symmetric(df):
	return np.allclose(df, df.T, equal_nan=True)


def binary(corr):
	log.started("Converting correlation matrix to pair-wise format.")
	if is_symmetric(corr):
		corr = convert_full_to_half_matrix(corr)

	stack = corr.stack().rename_axis(index=["gene1", "gene2"]).reset_index().rename(
		columns={0: "score"}
	)
	if has_mirror_of_first_pair(stack):
		log.info("Mirror pairs detected. Dropping them to ensure unique gene pairs.")
		stack = drop_mirror_pairs(stack)
	log.done("Pair-wise conversion.")
	return stack


def has_mirror_of_first_pair(df):
	g1, g2 = df.iloc[0]["gene1"], df.iloc[0]["gene2"]
	mirror_exists = ((df["gene1"] == g2) & (df["gene2"] == g1)).iloc[1:].any()
	return mirror_exists


def convert_full_to_half_matrix(df):
	if not is_symmetric(df):
		raise ValueError("Matrix must be symmetric to convert to half matrix.")

	log.started("Converting full correlation matrix to upper triangle (half-matrix) format.")
	arr = df.values.copy()
	arr[np.tril_indices_from(arr)] = np.nan
	log.done("Matrix conversion.")
	return pd.DataFrame(arr, index=df.index, columns=df.columns)


def drop_mirror_pairs(df):
	log.started("Dropping mirror pairs to ensure unique gene pairs (Optimized).")
	gene_pairs = np.sort(df[["gene1", "gene2"]].to_numpy(), axis=1)
	df.loc[:, ["gene1", "gene2"]] = gene_pairs
	df = df.loc[~df.duplicated(subset=["gene1", "gene2"], keep="first")]
	log.done("Mirror pairs are dropped.")
	return df


def quick_sort(df, ascending=False):
	log.started(f"Pair-wise matrix is sorting based on the 'score' column: ascending:{ascending}")
	order = 1 if ascending else -1
	sorted_df = df.iloc[np.argsort(order * df["score"].values)].reset_index(drop=True)
	log.done("Pair-wise matrix sorting.")
	return sorted_df


__all__ = [
	"perform_corr",
	"fast_corr",
	"is_symmetric",
	"binary",
	"has_mirror_of_first_pair",
	"convert_full_to_half_matrix",
	"drop_mirror_pairs",
	"quick_sort",
]
