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

    # select samples for this disease
    info_selected = info.query('Achilles_n_replicates.notnull() & primary_disease in @disease ')
    sample_names = list(set(info_selected.index) & set(df.columns))
    # use this samples and Seperate datasets and re-order big dataset, put small first
    d2 = df[sample_names] # small
    d1 = df.drop(sample_names,axis=1) # big dataset

    print(f"{disease} | shape of d1/d2: {d1.shape[1]} / {d2.shape[1]}")

    return d1, d2


def get_disease_corr_matrix(df, disease):
    
    disease_dir = os.path.join(result_dir, slugify.slugify(disease))
    os.makedirs(disease_dir, exist_ok=True)

    # if result file exists, return
    if  os.path.isfile(os.path.join(disease_dir,'x1.p')) and        os.path.isfile(os.path.join(disease_dir,'x2.p')):
        print(f'{disease} | correlation files are exist! It will be continued from file..')
        x1 = pd.read_parquet(os.path.join(disease_dir,'x1.p'))
        x2 = pd.read_parquet(os.path.join(disease_dir,'x2.p'))
        return x1, x2

    d1, d2 = get_separated_datasets(df, disease)

    print(f'{disease} | performing correlation, self correlations=np.nan ')

    # perform Pearson for X1 and X2, we do not need for X1X2 because it is not changed
    x1 = perform_corr(d1)
    x2 = perform_corr(d2)

    # save correlation results as a Parquet file because it is faster
    x1.to_parquet(os.path.join(disease_dir,'x1.p'))
    x2.to_parquet(os.path.join(disease_dir,'x2.p'))

    print(f'{disease} | correlation recently performed and files are saved. ')
    return x1, x2



def run_annotation_parallel(df,terms,n_core, n_split):
    stacks = np.array_split(df,n_split)
    results = Parallel(n_jobs=n_core)(delayed(annotate)(df,terms) for df in stacks)
    return pd.concat(results) # type: ignore


def perform_corr(df):
    corr = np.corrcoef(df.values)
    np.fill_diagonal(corr, np.nan)
    return pd.DataFrame(corr, index=df.index, columns=df.index)


def transfer_column(df_from,df_to,column_name):
    df_from = df_from.set_index(['gene1','gene2'])
    df_to = df_to.set_index(['gene1','gene2'])#.drop(f'{column_name}',axis=1)
    df_from = df_from[[column_name]]
    df = df_to.merge(df_from[column_name], how='outer', left_index=True, right_index=True, validate='one_to_one')
    return df.reset_index()


def read_terms(file):
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


# read splited csvs and concat
def merge_annotated_files(location):
    if location[-3:] == 'txt':
        files = glob.glob(location)
        arrs = [np.loadtxt(f) for f in files ]
        return np.concatenate(arrs)
    else:
        files = glob.glob(location)
        dfs = [pd.read_csv(f, index_col=0) for f in files]
        return pd.concat(dfs).reset_index(drop=True)


def convert_half_to_full_matrix(df):
    u = df.values
    l = df.T.values
    result = u.copy()
    l_indices = np.tril_indices_from(l)
    result[l_indices] = l[l_indices]
    np.fill_diagonal(result, np.nan)
    return pd.DataFrame(result, columns=df.columns,index=df.columns)


def convert_full_to_half_matrix(df):
    df.values[np.tril_indices(df.shape[0], -1)] = np.nan
    np.fill_diagonal(df.values, np.nan)
    return df





def get_mean_selected_genes(gene_set, corr_matrix, n_genes_threshold):
    check_genes = list(gene_set & set(corr_matrix.columns))
    if len(check_genes) < n_genes_threshold:
        pcc_mean = np.nan
    else:
        corr_matrix = corr_matrix.loc[check_genes,check_genes].copy()
        corr_binary = make_binary(corr_matrix)
        pcc_mean = corr_binary.score.mean()

    return pcc_mean


def annotate(df, terms):
    print(f'gene level annotation started..')
    df['annotated'] = df.apply(
        lambda x: sum(terms.set.map({x.gene1.upper(), x.gene2.upper()}.issubset)), axis=1
    )
    return df


def pr_analysis(df):

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


def makematrix(df):
    u = df.values
    l = df.T.values
    result = u.copy()
    l_indices = np.tril_indices_from(l)
    result[l_indices] = l[l_indices]
    np.fill_diagonal(result, np.nan)
    return pd.DataFrame(result, columns=df.columns,index=df.columns)

def make_binary(mat, remove_mirror=True, remove_self=True, sort=False):
    #print(f'remove_mirror:{remove_mirror}, remove_self:{remove_self}, sort:{sort}')
    stack = mat.copy().stack().reset_index()
    stack.columns = ['gene1', 'gene2','score']
    if remove_mirror:
        stack = drop_mirror_pairs(stack)
    if sort=='score':
        stack = stack.sort_values(['score'], ascending=False)
    # remove A,A or B,B
    if remove_self:
        stack = stack[stack.gene1 != stack.gene2]
    return stack


def drop_mirror_pairs(df):
    cols = ['gene1','gene2']
    df[cols] = np.sort(df[cols].values, axis=1)
    return df.drop_duplicates(subset=cols)





def fill_lower_matrix(df, fill=np.nan):
    ndf = df + df.T
    np.fill_diagonal(ndf.values, fill)
    return ndf


def projection(df1, df2):
    u, d, v = np.linalg.svd(df1, False)
    x1_projection = u @ u.T @ df2
    x1_projection.index = df1.index
    return x1_projection


def pra_single_complex(corr_matrix, common_genes, genes_in_single_complex, n_gene_threshold):

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

def perform_contribution(df):
# calculate correlation of each gene
# correlation contribution part
    mat = df.T
    K = len(mat.columns)
    corrm  = np.empty((K, K), dtype=float)
    corrm1 = np.empty((K, K), dtype=float)
    corrm2 = np.empty((K, K), dtype=float)
    contr1 = np.empty((K, K), dtype=float)
    contr2 = np.empty((K, K), dtype=float)
    denom = np.empty((K, K), dtype=float)

    for i in range(K):
        for j in range(K):
            if i > j:
                continue
            if i == j:
                pass
            else:
                x = mat.values[:,i]
                y = mat.values[:,j]
                r, r2, r1, x2_contr, x1_contr, denominator  = pcorr(x,y,7)
                corrm[i][j] = r
                corrm1[i][j] = r1
                corrm2[i][j] = r2
                contr1[i][j] = x1_contr
                contr2[i][j] = x2_contr
                denom[i][j] = denominator
    

    # save matrixes.
    pd.DataFrame(corrm,columns=df.T.columns,index=df.T.columns).to_parquet('corrm.parquet')
    pd.DataFrame(corrm1,columns=df.T.columns,index=df.T.columns).to_parquet('corrm1.parquet')
    pd.DataFrame(corrm2,columns=df.T.columns,index=df.T.columns).to_parquet('corrm2.parquet')
    pd.DataFrame(contr1,columns=df.T.columns,index=df.T.columns).to_parquet('contr1.parquet')
    pd.DataFrame(contr2,columns=df.T.columns,index=df.T.columns).to_parquet('contr2.parquet')
    pd.DataFrame(denom,columns=df.T.columns,index=df.T.columns).to_parquet('denom.parquet')

    return corrm, corrm1, corrm2, contr1, contr2, denom



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



def _pr_analysis_per_disease(df, common_genes, disease, annotated):
    
    result_dir = 'C:/Users/yd/Desktop/janus/results/per_disease/'
    disease_dir = os.path.join(result_dir, slugify.slugify(disease))
    os.makedirs(disease_dir, exist_ok=True)

    # if result file exists, return
    if  os.path.isfile(os.path.join(disease_dir,'pra1.p')) and os.path.isfile(os.path.join(disease_dir,'pra2.p')):
        pra1 = pd.read_parquet(os.path.join(disease_dir,'pra1.p'))
        pra2 = pd.read_parquet(os.path.join(disease_dir,'pra2.p'))
        return pra1, pra2
        

    # select samples for this disease
    info_selected = info.query('Achilles_n_replicates.notnull() & primary_disease in @disease ')
    sample_names = list(set(info_selected.index) & set(df.columns))


    # use this samples and Seperate datasets and re-order big dataset, put small first
    d1 = df.drop(sample_names,axis=1) # big dataset
    d2 = df[sample_names] # small

    # perform Pearson for X1 and X2, we do not need for X1X2 because it is not changed
    x1 = perform_corr(d1)
    x2 = perform_corr(d2)

    # save correlation results as a Parquet file because it is faster
    x1.to_parquet(os.path.join(disease_dir,'x1.p'))
    x2.to_parquet(os.path.join(disease_dir,'x2.p'))

    # filter matrix using common genes from TermsDB
    f1 = x1.loc[common_genes,common_genes].copy()
    f2 = x2.loc[common_genes,common_genes].copy()

    convert_full_to_half_matrix(f1)
    convert_full_to_half_matrix(f2)
   
    s1 = make_binary(f1)
    s2 = make_binary(f2)
    
    # transfer annotation column from X12 to X1 and X2
    annotated1 = transfer_column(annotated, s1, 'annotated')
    annotated2 = transfer_column(annotated, s2, 'annotated')

    # run PR analysis
    pra1 = pr_curve_analysis(annotated1)
    pra2 = pr_curve_analysis(annotated2)

    # save PR result
    pra1.to_parquet(os.path.join(disease_dir,'pra1.p'))
    pra2.to_parquet(os.path.join(disease_dir,'pra2.p'))    

    return pra1, pra2




        
def __pra_per_complex_parallel(df, terms, common_genes, disease):
    
    def _f(X):
        return terms.iloc[:3].set.progress_apply(lambda g: pra_per_complex(X, common_genes, g, 3))

    x1, x2 = get_disease_corr_matrix(df, disease)
    results = Parallel(n_jobs=2,verbose=3)(delayed(_f)(X) for X in [x1,x2])

    auc1, _ = list(zip(*results[0])) #type: ignore
    auc2, _ = list(zip(*results[1])) #type: ignore

    terms['auc1'] = auc1
    terms['auc2'] = auc2
    terms = terms.drop(['ID','list','set'],axis=1)
    terms.to_parquet(os.path.join(result_dir, slugify.slugify(disease) ,'auc1_auc2.p'))

    print(f'{disease}: DONE.')

    return {disease: [auc1, auc2]}





def partial_corr(df, separate):

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


