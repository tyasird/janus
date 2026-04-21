import os
from importlib import resources

import pandas as pd
from tqdm import tqdm

from .logging_config import log
from .utils import dsave, dload

tqdm.pandas()


# def get_example_data_path(filename: str):
#     return resources.files("benchmarkcr.data").joinpath("dataset").joinpath(filename)


def _load_file(filepath, ext):
    loaders = {
        ".csv": lambda f: pd.read_csv(f, index_col=0),
        ".xlsx": lambda f: pd.read_excel(f, index_col=0),
        ".parquet": pd.read_parquet,
        ".p": pd.read_parquet
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file extension: {ext}")

    return loaders[ext](filepath)


def load_datasets(files):
    preprocessing = dload("config")["preprocessing"]
    data_dict = {}

    for filename, meta in files.items():
        if isinstance(meta, pd.DataFrame):
            df = meta
        elif isinstance(meta, dict):
            filepath = meta["path"]
            if isinstance(filepath, pd.DataFrame):
                df = filepath
            else:
                ext = os.path.splitext(filepath)[1]
                df = _load_file(filepath, ext)
        else:
            raise ValueError(f"Unsupported data structure for '{filename}': {type(meta)}")

        df.index = df.index.str.split().str[0]

        if preprocessing.get('drop_na'):
            log.info(f"{filename}: Dropping missing values.")
            df = df.dropna(axis=0)

        fill_na = preprocessing.get('fill_na')
        if fill_na == 'mean':
            log.info(f"{filename}: Filling missing values with column mean.")
            df = df.T.fillna(df.mean(axis=1)).T

        if fill_na == 'zero':
            log.info(f"{filename}: Filling missing values with zeros.")
            df = df.fillna(0)

        data_dict[filename] = df

    common_genes = get_common_genes(data_dict)
    log.info(f"Continuing with common genes: {len(common_genes)}")
    for filename, df in data_dict.items():
        if df.index.isin(common_genes).any():
            data_dict[filename] = df.loc[common_genes]

    dsave({
        "datasets": data_dict,
        "sorting": {
            k: v.get("sort", "high") if isinstance(v, dict) else "high"
            for k, v in files.items()
        }
    }, "input")
    log.done("Datasets loaded.")
    return data_dict, common_genes


def get_common_genes(datasets):
    log.started("Finding common genes across datasets.")
    gene_sets = [set(df.index) for df in datasets.values()]
    common_genes = list(set.intersection(*gene_sets))
    log.done(f"Common genes found: {len(common_genes)}")
    dsave(common_genes, "tmp", "common_genes")
    return common_genes


def merge_datasets():
    data = dload("input")
    common_genes = dload("tmp", "common_genes")
    if not common_genes:
        raise ValueError("No common genes found. Please load datasets first.")
    data_dict = data["datasets"]
    common_genes = get_common_genes(data_dict)

    dataset_names = list(data_dict.keys())
    if len(dataset_names) != 2:
        raise ValueError("Exactly two datasets expected.")

    df1 = data_dict[dataset_names[0]].loc[common_genes]
    df2 = data_dict[dataset_names[1]].loc[common_genes]

    merged_df = pd.concat([df1, df2], axis=1)
    first_dataset_sample_count = df1.shape[1]

    return merged_df, first_dataset_sample_count