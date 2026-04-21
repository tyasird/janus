import argparse
from pathlib import Path

from janusbio.analysis import (
    get_contributions,
    get_covariation_contributions,
    get_differential_correlation,
)
from janusbio import preprocessing as pp
from janusbio import utils


def _save_matrix(df, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / filename)


def build_parser():
    parser = argparse.ArgumentParser(description="JANUS CLI for contribution-based analyses")
    parser.add_argument("--ref", required=True, help="Path to reference dataset")
    parser.add_argument("--exp", required=True, help="Path to experiment dataset")
    parser.add_argument(
        "--pairwise_contribution",
        action="store_true",
        help="Run get_contributions on merged ref+exp data",
    )
    parser.add_argument(
        "--get_covariation_contributions",
        action="store_true",
        help="Run covariation contribution calculation",
    )
    parser.add_argument(
        "--get_differential_correlation",
        action="store_true",
        help="Run differential correlation calculation",
    )
    parser.add_argument(
        "--output",
        help="Optional output directory. If provided, results are saved as parquet files.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    selected_formulas = [
        args.pairwise_contribution,
        args.get_covariation_contributions,
        args.get_differential_correlation,
    ]
    if not any(selected_formulas):
        parser.error("Select at least one formula flag.")

    utils.initialize(
        {
            "preprocessing": {
                "drop_na": True,
            }
        }
    )

    inputs = {
        "REFERENCE": {"path": args.ref, "sort": "high"},
        "EXPERIMENT": {"path": args.exp, "sort": "high"},
    }

    pp.load_datasets(inputs)
    try:
        merged_df, reference_sample_size = pp.merge_datasets()
    except ValueError as exc:
        if "No common genes found" in str(exc):
            raise ValueError(
                "CLI expects raw reference/experiment datasets with genes as rows and samples as columns. "
                "Do not pass precomputed contribution/correlation matrices like reference_contr.parquet or experiment_contr.parquet."
            ) from exc
        raise

    output_dir = Path(args.output) if args.output else None

    if args.pairwise_contribution:
        reference_contr, experiment_contr = get_contributions(
            merged_df,
            reference_sample_size=reference_sample_size,
        )
        print("Computed pairwise contribution matrices.")
        if output_dir is not None:
            _save_matrix(reference_contr, output_dir, "pairwise_reference_contribution.parquet")
            _save_matrix(experiment_contr, output_dir, "pairwise_experiment_contribution.parquet")

    if args.get_covariation_contributions:
        reference_cov, experiment_cov = get_covariation_contributions(
            merged_df,
            reference_sample_size=reference_sample_size,
        )
        print("Computed covariation contribution matrices.")
        if output_dir is not None:
            _save_matrix(reference_cov, output_dir, "covariation_reference_contribution.parquet")
            _save_matrix(experiment_cov, output_dir, "covariation_experiment_contribution.parquet")

    if args.get_differential_correlation:
        diff_corr, direction = get_differential_correlation(
            merged_df,
            reference_sample_size=reference_sample_size,
        )
        print("Computed differential correlation matrices.")
        if output_dir is not None:
            _save_matrix(diff_corr, output_dir, "differential_correlation.parquet")
            _save_matrix(direction, output_dir, "differential_correlation_direction.parquet")

    if output_dir is not None:
        print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()