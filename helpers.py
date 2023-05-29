import glob
import pandas as pd
import numpy as np
from IPython import embed
from sklearn import metrics
from joblib import Parallel,delayed
import os
import slugify
from tqdm.notebook import tqdm
tqdm.pandas()

#%%
info = pd.read_csv('./input/Achilles_gene_effect_20Q2_info.csv',index_col=0)
result_dir = 'C:/Users/yd/Desktop/janus/results/per_disease/'


def get_separated_datasets(df, disease):
    """
    Separates a given DataFrame into two datasets based on a specified disease.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        disease (str or list): The disease(s) used to filter the data.

    Returns:
        tuple: A tuple containing two DataFrames. The first DataFrame (d1) contains the columns
               that do not match the specified disease, and the second DataFrame (d2) contains
               the columns that match the specified disease.
    """
    info_selected = info.query('Achilles_n_replicates.notnull() & primary_disease in @disease ')
    sample_names = list(set(info_selected.index) & set(df.columns))
    # use this samples and Seperate datasets and re-order big dataset, put small first
    d2 = df[sample_names] # small
    d1 = df.drop(sample_names,axis=1) # big dataset
    print(f"{disease} | shape of d1/d2: {d1.shape[1]} / {d2.shape[1]}")
    return d1, d2


def get_disease_corr_matrix(df, disease):
    """
    Retrieves or computes the correlation matrix for a specified disease in a given DataFrame.

    If the correlation matrix files already exist for the specified disease, the function will retrieve them. Otherwise, it will compute the correlation matrix by separating the DataFrame into two datasets based on the disease and performing Pearson correlation on each dataset.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        disease (str): The disease used to filter the data and compute the correlation matrix.

    Returns:
        tuple: A tuple containing two DataFrames. The first DataFrame (x1) represents the correlation matrix for the columns that do not match the specified disease. The second DataFrame (x2) represents the correlation matrix for the columns that match the specified disease.

    """

    disease_dir = os.path.join(result_dir, slugify.slugify(disease))
    os.makedirs(disease_dir, exist_ok=True)
    # if result file exists, return from file
    if  os.path.isfile(os.path.join(disease_dir,'x1.p')) and        os.path.isfile(os.path.join(disease_dir,'x2.p')):
        print(f'{disease} | correlation files are exist! It will be continued from file..')
        x1 = pd.read_parquet(os.path.join(disease_dir,'x1.p'))
        x2 = pd.read_parquet(os.path.join(disease_dir,'x2.p'))
        return x1, x2

    d1, d2 = get_separated_datasets(df, disease)
    print(f'{disease} | performing correlation, self correlations=np.nan ')
    # perform Pearson for X1 and X2
    x1 = perform_corr(d1)
    x2 = perform_corr(d2)
    x1.to_parquet(os.path.join(disease_dir,'x1.p'))
    x2.to_parquet(os.path.join(disease_dir,'x2.p'))
    print(f'{disease} | correlation recently performed and files are saved. ')
    return x1, x2



def run_annotation_parallel(df,terms,n_core, n_split):
    """
    Runs parallel annotation of a DataFrame using a specified number of cores and splits.

    The function divides the input DataFrame into multiple stacks based on the specified number of splits, and performs annotation in parallel using the specified number of cores. It utilizes the `annotate` function to perform the annotation on each stack. The results from each stack are concatenated into a single DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame to be annotated.
        terms (list): The annotation terms or keywords to use for the annotation.
        n_core (int): The number of cores to use for parallel annotation.
        n_split (int): The number of splits to divide the DataFrame into.

    Returns:
        pandas.DataFrame: A DataFrame containing the annotated results from all stacks.

    """
    stacks = np.array_split(df,n_split)
    results = Parallel(n_jobs=n_core)(delayed(annotate)(df,terms) for df in stacks)
    return pd.concat(results) # type: ignore


def perform_corr(df):
    """
    Computes the correlation matrix for a given DataFrame.

    The function calculates the correlation matrix using the numpy `corrcoef` function and sets the diagonal elements to NaN to avoid self-correlation. The resulting correlation matrix is returned as a DataFrame with the same index and columns as the input DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame for which the correlation matrix is computed.

    Returns:
        pandas.DataFrame: The correlation matrix of the input DataFrame.

    """
    corr = np.corrcoef(df.values)
    np.fill_diagonal(corr, np.nan)
    return pd.DataFrame(corr, index=df.index, columns=df.index)


def transfer_column(df_from,df_to,column_name):
    """
    Transfers a specific column from one DataFrame to another based on common index columns.
    The function sets the index of both input DataFrames to ['gene1', 'gene2'] and selects the specified column from the 'df_from' DataFrame. It then merges the selected column with the 'df_to' DataFramebase d on the common index columns in an outer join. The resulting DataFrame is reset with the default index and returned.

    Args:
        df_from (pandas.DataFrame): The source DataFrame from which the column is transferred.
        df_to (pandas.DataFrame): The target DataFrame to which the column is transferred.
        column_name (str): The name of the column to transfer.

    Returns:
        pandas.DataFrame: The 'df_to' DataFrame with the transferred column included.

    """
    df_from = df_from.set_index(['gene1','gene2'])
    df_to = df_to.set_index(['gene1','gene2'])#.drop(f'{column_name}',axis=1)
    df_from = df_from[[column_name]]
    df = df_to.merge(df_from[column_name], how='outer', left_index=True, right_index=True, validate='one_to_one')
    return df.reset_index()


def read_terms(file):
    """
    Reads and processes a terms file, returning the terms DataFrame and unique genes.

    The function reads a terms file as a pandas DataFrame and performs several
    data processing steps on the DataFrame, including filtering out rows with Length equal to 1,
    replacing certain characters in the 'Genes' column, splitting the 'Genes' column into a list,
    and creating a set of genes for each row. The function also computes the unique genes from the
    'list' column to be used for filtering the correlation matrix. Finally, the processed terms
    DataFrame and unique genes are returned.

    Args:
        file (str): The path to the terms file.

    Returns:
        tuple: A tuple containing the processed terms DataFrame and the unique genes as a numpy array.

    """
    terms = pd.read_csv(file, index_col=0)
    terms = terms[terms.Length != 1]
    terms['Genes'] = terms.Genes.str.replace(';;',';')
    terms['Genes'] = terms.Genes.str.replace(' ','')
    terms['Genes'] = terms.Genes.apply(lambda x: x[:-1] if x[-1] == ';' else x)
    terms["list"] = terms.apply(lambda x: x["Genes"].split(";"), axis=1)
    terms['set'] = terms['Genes'].apply(lambda x: set(x.split(';')))
    # We will use this uniq gene list to filter correlation matrix
    # Otherwise, when we check gene pairs in the complex, it will give us 0
    terms_uniq_genes = np.unique(terms['list'].explode().values) # type: ignore
    # terms and dataset common genes
    terms = terms.reset_index(drop=True)
    return terms, terms_uniq_genes




def convert_half_to_full_matrix(df):
    """
    Converts a half-matrix DataFrame to a full-matrix DataFrame. The function takes a half-matrix DataFrame as input, where the upper triangular part represents the values, and the lower triangular part is filled with NaN values. It creates a copy of the upper triangular values and replaces the lower triangular values with the corresponding values from the transpose of the DataFrame. The resulting full-matrix DataFrame has NaN values in the diagonal. The column and index labels are preserved from the input DataFrame.

    Args:
        df (pandas.DataFrame): The half-matrix DataFrame to convert to a full-matrix.

    Returns:
        pandas.DataFrame: The full-matrix DataFrame.

    """
    u = df.values
    l = df.T.values
    result = u.copy()
    l_indices = np.tril_indices_from(l)
    result[l_indices] = l[l_indices]
    np.fill_diagonal(result, np.nan)
    return pd.DataFrame(result, columns=df.columns,index=df.columns)


def convert_full_to_half_matrix(df):
    """
    Converts a half-matrix DataFrame to a full-matrix DataFrame. The resulting full-matrix DataFrame has NaN values in the diagonal. The column and index labels are preserved from the input DataFrame.

    Args:
        df (pandas.DataFrame): The half-matrix DataFrame to convert to a full-matrix.

    Returns:
        pandas.DataFrame: The full-matrix DataFrame.

    """
    df.values[np.tril_indices(df.shape[0], -1)] = np.nan
    np.fill_diagonal(df.values, np.nan)
    return df



def get_mean_selected_genes(gene_set, corr_matrix, n_genes_threshold):
    """
    Calculates the mean score of the binary correlation matrix for the selected genes.

    The function takes a gene set, a correlation matrix, and a threshold for the minimum number of genes required for calculation. It checks if the number of genes in the intersection of the gene set and the correlation matrix columns is below the threshold. If so, it returns NaN. Otherwise, it selects the corresponding submatrix from the correlation matrix, converts it into a binary matrix using the `make_binary` function, and computes the mean score.
    The mean score is returned as the result.

    Args:
        gene_set (set): The set of genes to select from the correlation matrix.
        corr_matrix (pandas.DataFrame): The correlation matrix containing all genes.
        n_genes_threshold (int): The minimum number of genes required for calculation.

    Returns:
        float: The mean score of the binary correlation matrix for the selected genes, or NaN if the threshold is not met.
    """
    check_genes = list(gene_set & set(corr_matrix.columns))
    if len(check_genes) < n_genes_threshold:
        pcc_mean = np.nan
    else:
        corr_matrix = corr_matrix.loc[check_genes,check_genes].copy()
        corr_binary = make_binary(corr_matrix)
        pcc_mean = corr_binary.score.mean()

    return pcc_mean


def annotate(df, terms):
    """
    Performs gene level annotation on a DataFrame using a set of terms.

    The function adds an 'annotated' column to the input DataFrame, which indicates the level of annotation for each gene pair. It applies a lambda function to each row of the DataFrame, checking if the combination  of 'gene1' and 'gene2' exists as a subset in the 'terms' set. The sum of True values is computed as the annotation level and assigned to the 'annotated' column. The annotated DataFrame is returned.

    Args:
        df (pandas.DataFrame): The DataFrame to annotate.
        terms (set): The set of terms used for gene annotation.

    Returns:
        pandas.DataFrame: The annotated DataFrame with the 'annotated' column added.
    """
    print(f'gene level annotation started..')
    df['annotated'] = df.apply(
        lambda x: sum(terms.set.map({x.gene1.upper(), x.gene2.upper()}.issubset)), axis=1
    )
    return df


def pr_analysis(df):
    """
    Performs precision-recall analysis on a DataFrame.

    Sorts the DataFrame by the 'score' column in descending order. Calculates the 'prediction' column based on the presence of annotation in the 'annotated' column. Computes the true positives ('tp') column by cumulative summation of the 'prediction' column. Resets the index and calculates the precision and recall values based on the 'tp' column and the row index. Returns the DataFrame with the added 'precision' and 'recall' columns.
    """

    # sort values by pcc
    df = df.sort_values(by=["score"], ascending=False)

    # calculate prediction, if it is annotated set 1
    df["prediction"] = df.annotated.apply(lambda x:1 if x !=0 else 0 )

    # calculate TP by row order
    df["tp"] = df.prediction.cumsum()

    # reset index, we will use this index for calculation of precision
    df = df.reset_index().drop("index", axis=1)

    # calculate precision and FP
    #x1_an["precision"] = x1_an.apply(lambda x: int(x.tp) / (int(x["index"]) + 1), axis=1)
    #x1_an["fp"] = x1_an.apply(lambda x: x.tp / x1_an.iloc[-1].tp, axis=1)
    df['precision'] = df.tp/(df.index+1)
    df['recall'] = df.tp/df.iloc[-1].tp   

    return df


def pr_analysis_quick(corr_matrix, common_genes, annotated, disease, check_file):
    """
        This function copies annotation from big dataset to other dataset
        Quick version of pr_analysis
    """
    print(f'{disease} | quick annotation and PRA is started..')
    result_dir = 'C:/Users/yd/Desktop/janus/results/per_disease/'
    disease_dir = os.path.join(result_dir, slugify.slugify(disease))
    os.makedirs(disease_dir, exist_ok=True)

    # if result file exists, return
    if  os.path.isfile(os.path.join(disease_dir, check_file)):
        print(f'{disease} | PRA is exist, It will be continued from file.. ')
        pra = pd.read_parquet(os.path.join(disease_dir,check_file))
        return pra
        
    # filter matrix using common genes from TermsDB
    f = corr_matrix.loc[common_genes,common_genes].copy()
    convert_full_to_half_matrix(f)
    s = make_binary(f)
    # transfer annotation column from X12 to X
    annotated = transfer_column(annotated, s, 'annotated')
    # run PR analysis
    pra = pr_analysis(annotated)
    # save PR result
    pra.to_parquet(os.path.join(disease_dir,check_file))
    print(f'{disease} | PRA result is saved.')
    return pra



def make_binary(mat, remove_mirror=True, remove_self=True, sort=False):
    """
    Converts a matrix into a binary representation.

    Transforms the input matrix into a binary representation by stacking the values with the corresponding gene pairs.  If 'remove_mirror' is set to True, mirror pairs (e.g., (A, B) and (B, A)) are removed. If 'remove_self' is set to True, pairs with the same gene (e.g., (A, A) and (B, B)) are removed. If 'sort' is set to 'score', the resulting DataFrame is sorted by the 'score' column in descending order. The transformed DataFrame is returned.
    """
    #print(f'remove_mirror:{remove_mirror}, remove_self:{remove_self}, sort:{sort}')
    stack = mat.copy().stack().reset_index()
    stack.columns = ['gene1', 'gene2','score']
    if remove_mirror:
        stack = drop_mirror_pairs(stack)
    if sort=='score':
        stack = stack.sort_values(['score'], ascending=False)
    if remove_self:
        stack = stack[stack.gene1 != stack.gene2]
    return stack


def drop_mirror_pairs(df):
    """
    Drops mirror pairs from a DataFrame.
    """
    cols = ['gene1','gene2']
    df[cols] = np.sort(df[cols].values, axis=1)
    return df.drop_duplicates(subset=cols)



def pra_single_complex(corr_matrix, common_genes, genes_in_single_complex, n_gene_threshold):
    """
    Performs precision-recall analysis for a single gene complex.

    Selects the submatrix from the correlation matrix based on the common genes. If the correlation matrix is in triangular form, it is converted to a full matrix. Checks if the intersection of the 'genes_in_single_complex' set and the selected genes is above the threshold. If below the threshold, returns NaN values for the score and  the result DataFrame. Otherwise, creates a DataFrame with the selected genes and performs precision-recall
    analysis using the 'pr_analysis' function. Calculates the area under the precision-recall curve (AUC) as the score. Returns the score and the number of selected genes in a list.

    Args:
        corr_matrix (pandas.DataFrame): The correlation matrix.
        common_genes (list): List of common genes.
        genes_in_single_complex (set): Set of genes in the single complex.
        n_gene_threshold (int): Threshold for the number of genes in the complex.

    Returns:
        list: A list containing the score (area under the precision-recall curve) and the number of selected genes.

    """
    corr_matrix = corr_matrix.loc[common_genes,common_genes].copy()
    # if it is triangle matrix, convert to full matrix
    if np.isnan(corr_matrix.iloc[-1,0]):
        corr_matrix = convert_half_to_full_matrix(corr_matrix)

    checkgenes = list(genes_in_single_complex & set(corr_matrix.columns))
    if len(checkgenes) < n_gene_threshold:
        score = np.nan
        df = np.nan
    else:
        fmatrix = corr_matrix.loc[checkgenes, list(set(corr_matrix.columns))].copy()
        df = fmatrix.stack().reset_index()
        df.columns = ['gene1', 'gene2','score']
        df = drop_mirror_pairs(df)
        df['annotated'] = df.apply(
            lambda x:  1 if genes_in_single_complex.issuperset( {x.gene1,x.gene2} ) else 0, axis=1
        )
        df = pr_analysis(df)
        score = metrics.auc(df.recall, df.precision)
    return [score, len(checkgenes)]

      
def pra_single_complex_parallel(df, terms, common_genes, disease):
    """
    Performs parallel precision-recall analysis for multiple gene complexes.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        terms (pandas.DataFrame): DataFrame containing the gene complexes and associated information.
        common_genes (list): List of common genes.
        disease (str): The name of the disease.

    Returns:
        pandas.DataFrame: Updated DataFrame with the calculated AUC values for each gene complex.
    """
    x1, x2 = get_disease_corr_matrix(df, disease)

    def _f(X):
        return terms.set.progress_apply(lambda g: pra_single_complex(X, common_genes, g, 3))

    r = Parallel(n_jobs=2)(delayed(_f)(X) for X in tqdm([x1,x2]))
    score, n_gene_used = list(zip(*r))

    terms['auc1'] = np.nan
    terms['auc2'] = np.nan
    terms = terms.drop(['ID','list','set'],axis=1)
    terms.to_parquet(os.path.join(result_dir, slugify.slugify(disease) ,'auc1_auc2.p'))

    return terms


#####################################################

def pcorr(x,y,column): 
    # x = gene1 vector
    # y = gene2 vector
    # column seperates dataset i.e: 7

    # sample seperation (A and B) for gene 1
    xA,xB = x[:column], x[column:]
    # sample seperation (A and B) for gene 1
    yA,yB = y[:column], y[column:]

    # means
    x_mean, y_mean = x.mean(), y.mean()

    # mean distance  each sample group A and B for gene1
    xAm, xBm = xA - x_mean, xB - x_mean
    # mean distance  each sample group A and B for gene2
    yAm, yBm = yA - y_mean, yB - y_mean

    # mean distance for vector gene1 and gene2    
    xm, ym = x - x_mean, y - y_mean

    # np.linalg.norm is the L2-Norm: calculates Euclidean distance for given vector 
    # we will use this value as a denominator
    normxm = np.linalg.norm(xm)
    normym = np.linalg.norm(ym)

    # np.dot is matrix multiplication
    # multiplicates A and B vectors and divedes to L2=norm value (A*B)
    denom = (normxm*normym)
    small_contr_val = np.dot(xAm,yAm)
    big_contr_val =  np.dot(xBm,yBm)
    r = np.dot(xm,ym) / denom
    small_contr = small_contr_val / denom # small contr.
    big_contr = big_contr_val / denom # big contr.


    return r, small_contr, big_contr, small_contr_val,  big_contr_val,  denom




def partial_corr(df, separate):
    """
    Calculates partial correlation matrices based on the input DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        separate (int): The number of columns to separate in the DataFrame.

    Returns:
        tuple: A tuple containing the following partial correlation matrices:
            - contr_big: Partial correlation matrix for the larger subset.
            - contr_small: Partial correlation matrix for the smaller subset.
            - m1: Cross-product matrix for the larger subset.
            - m2: Cross-product matrix for the smaller subset.
            - denom: Denominator matrix for the partial correlation calculations.
    """
    cov = df.T.cov()
    std = df.T.std(ddof=1)
    std = pd.DataFrame(np.outer(std,std),columns=df.index, index=df.index)

    global_mean = df.T.mean()
    d12_mean = df.apply(lambda x: x - global_mean)
    d1_mean = d12_mean.iloc[:, separate:]
    d2_mean = d12_mean.iloc[:, :separate]

    m1 = pd.DataFrame(np.dot(d1_mean,d1_mean.T),columns=df.index,index=df.index)
    np.fill_diagonal(m1.values, 0)
    m2 = pd.DataFrame(np.dot(d2_mean,d2_mean.T),columns=df.index,index=df.index)
    np.fill_diagonal(m2.values, 0)

    denom = d12_mean.T.apply(np.linalg.norm)
    denom = pd.DataFrame(np.outer(denom,denom.T),columns=df.index,index=df.index)
    np.fill_diagonal(denom.values,0)

    contr_big = m1/denom    
    contr_small = m2/denom   
    #corr = cov/std 

    for i in [contr_big, contr_small, m1, m2, denom]:
        convert_full_to_half_matrix(i)
        np.fill_diagonal(i.values, np.nan)

    return contr_big, contr_small, m1, m2, denom


