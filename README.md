# JANUS CLI

This project can be run from the command line with `janus.py` or `python -m janusbio.cli`.

## Installation



### Install from GitHub

```bash
pip install git+https://github.com/tyasird/janus.git
```

### Install from PyPI

```bash
pip install janusbio
```

### Editable install for development

```bash
pip install -e .
```

After installation, you can run the CLI as:

```bash
janus --help
```

## Input format

- Provide 2 raw datasets:
  - `--ref`: reference dataset
  - `--exp`: experiment dataset
- Expected format for both files:
  - genes as rows
  - samples as columns
- Supported file types:
  - `.csv`
  - `.xlsx`
  - `.parquet`
  - `.p`

Do **not** pass precomputed contribution or correlation matrices.

## Basic usage

Run one or more formulas from the command line:

```bash
python janus.py \
  --ref depmap_23Q2_chronos.csv \
  --exp mito_screens.csv \
  --pairwise_contribution \
  --get_covariation_contributions \
  --get_differential_correlation
```

Equivalent module form:

```bash
python -m janusbio.cli \
  --ref depmap_23Q2_chronos.csv \
  --exp mito_screens.csv \
  --pairwise_contribution
```

## Available flags

### Required inputs

- `--ref <path>`: path to the reference dataset
- `--exp <path>`: path to the experiment dataset

### Formula selection

- `--pairwise_contribution`
  - runs `get_contributions(...)`
  - produces reference and experiment contribution matrices

- `--get_covariation_contributions`
  - runs `get_covariation_contributions(...)`
  - produces reference and experiment covariation contribution matrices

- `--get_differential_correlation`
  - runs `get_differential_correlation(...)`
  - produces differential-correlation and direction matrices

At least one formula flag must be provided.

### Optional output

- `--output <directory>`
  - if provided, results are saved as parquet files

Example:

```bash
python janus.py \
  --ref depmap_23Q2_chronos.csv \
  --exp mito_screens.csv \
  --pairwise_contribution \
  --get_covariation_contributions \
  --get_differential_correlation \
  --output output
```

## Output files

Depending on the selected flags, the CLI writes:

- `pairwise_reference_contribution.parquet`
- `pairwise_experiment_contribution.parquet`
- `covariation_reference_contribution.parquet`
- `covariation_experiment_contribution.parquet`
- `differential_correlation.parquet`
- `differential_correlation_direction.parquet`

## Help

To see CLI help:

```bash
python janus.py --help
```

Or:

```bash
python -m janusbio.cli --help
```