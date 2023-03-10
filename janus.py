#%%
import pandas as pd
from sklearn import preprocessing
from scipy.stats import pearsonr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from scipy.linalg import svd

# def partialcorr(x,y, x_mean, y_mean):
#     xm = x - x_mean
#     ym = y - y_mean
#     normxm = np.linalg.norm(xm)
#     normym = np.linalg.norm(ym)
#     xm_ym_multiple = np.dot(xm,ym)
#     norm = normxm*normym
#     return xm, ym, norm, xm_ym_multiple

#%%
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
    r = np.dot(xm,ym) / (normxm*normym)
    r1 = np.dot(xAm,yAm) / (normxm*normym) # small contr.
    r2 = np.dot(xBm,yBm) / (normxm*normym) # big contr.

    return r, r1, r2, np.dot(xAm,yAm),  np.dot(xBm,yBm),  (normxm*normym)


def makematrix(df):
    u = df.values
    l = df.T.values
    result = u.copy()
    l_indices = np.tril_indices_from(l)
    result[l_indices] = l[l_indices]
    np.fill_diagonal(result, np.nan)
    return pd.DataFrame(result, columns=df.columns,index=df.columns)


def mat_to_binary(mat):
    stack = mat.stack().reset_index()
    stack.columns = ['gene1', 'gene2','score']
    stack = stack[stack.gene1 < stack.gene2]
    return stack


def projection(df1, df2):
    u, d, v = np.linalg.svd(df1, False)
    x1_projection = u @ u.T @ df2
    x1_projection.index = df1.index
    return x1_projection


# %%
# read 11kgenes and crispr genes
df1 = pd.read_csv('./data/crisprgeneeffects.csv', index_col=0).T
df2 = pd.read_csv('./data/11kgenes.csv', index_col=0)
# delete paranteses from gene names
df1.index = [i.split(' ')[0] for i in df1.index]
common_genes = list(set(df2.index) & set(df1.index))
# filter only common genes
df1 = df1.loc[common_genes]
df2 = df2.loc[common_genes]
# zscore norm. for only X1
scaled = (df1 - df1.mean())/df1.std(ddof=0)
df1_scaled = pd.DataFrame(scaled, index=df1.index, columns=df1.columns)
# center only X1
df1_centered = df1_scaled.apply(lambda x: x-x.mean(),axis=1)
# concat X1 and X2
dfc = pd.concat([df2,df1_centered], axis=1)

d1 = dfc.iloc[:,7:] # big dataset
d2 = dfc.iloc[:,:7] # small


# %%
# calculate correlation of each gene
mat = dfc.T
K = len(mat.columns)
corrm  = np.empty((K, K), dtype=float)
corrm1 = np.empty((K, K), dtype=float)
corrm2 = np.empty((K, K), dtype=float)
x1_contr_m = np.empty((K, K), dtype=float)
x2_contr_m = np.empty((K, K), dtype=float)
denominator_m = np.empty((K, K), dtype=float)

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
            x1_contr_m[i][j] = x1_contr
            x2_contr_m[i][j] = x2_contr
            denominator_m[i][j] = denominator

pd.DataFrame(corrm,columns=mat.columns,index=mat.columns).to_csv('corrm.csv')
pd.DataFrame(corrm1,columns=mat.columns,index=mat.columns).to_csv('corrm1.csv')
pd.DataFrame(corrm2,columns=mat.columns,index=mat.columns).to_csv('corrm2.csv')
pd.DataFrame(x1_contr_m,columns=mat.columns,index=mat.columns).to_csv('x1_contr_m.csv')
pd.DataFrame(x2_contr_m,columns=mat.columns,index=mat.columns).to_csv('x2_contr_m.csv')
pd.DataFrame(denominator_m,columns=mat.columns,index=mat.columns).to_csv('denominator_m.csv')


#%%
# Contribution matrixes
corrm = pd.read_csv('./corrm.csv', index_col=0)
corrm1 = pd.read_csv('./corrm1.csv', index_col=0)
corrm2 = pd.read_csv('./corrm2.csv', index_col=0)
x1_contr_ints = pd.read_csv('./x1_contr_m.csv', index_col=0)
x2_contr_ints = pd.read_csv('./x2_contr_m.csv', index_col=0)
denominator_m = pd.read_csv('./denominator_m.csv', index_col=0)

# Main Correlations
x1_corr = d1.T.corr() # big corr.
x2_corr = d2.T.corr() # small corr.
x1x2_corr = dfc.T.corr() # merged corr.
np.fill_diagonal(x1_corr.values, np.nan)
np.fill_diagonal(x2_corr.values, np.nan)
np.fill_diagonal(x1x2_corr.values, np.nan)

# Column order for main correlations
x1_corr = x1_corr.loc[corrm.columns, corrm.columns].copy()
x2_corr = x2_corr.loc[corrm.columns, corrm.columns].copy()
x1x2_corr = x1x2_corr.loc[corrm.columns, corrm.columns].copy()


# %%
# stack contribituon scores for finding extreme values
stack = mat_to_binary(corrm)
stack1 = mat_to_binary(corrm1)
stack2 = mat_to_binary(corrm2)

# extreme gene pairs from small contribituon dataset
extreme_pairs = stack2.query('score>0.29').sort_values('score')[-5000:]
pairs = extreme_pairs[['gene1','gene2']].copy().reset_index(drop=True)\
.sort_values(by=['gene1', 'gene2'])
pairstuple = zip(pairs.gene1,pairs.gene2)


#%%
df3 = pd.read_excel('./data/11kgeneswithDMSO.xlsx', index_col=0)
df3 = df3.loc[common_genes]
x3_corr = df3.T.corr()
x3_corr = x3_corr.loc[corrm.columns,corrm.columns]
np.fill_diagonal(x3_corr.values, np.nan)


#%%
# for each gene pair fetch scores from different matrices
arr = [x1_corr, x2_corr, x3_corr, x1x2_corr, corrm1, corrm2, x1_contr_ints, x2_contr_ints, denominator_m]
arr_features = ['BIG_corr', 'SMALL_corr','SMALL_with_control_corr', 'MERGED_corr', 'BIG_contribution', 'SMALL_contribution', 'BIG_contr_without_denominator', 'SMALL_contr_without_denominator', 'denominator']
new_arr = []

for k,v in enumerate(arr):
    pairstuple = zip(pairs.gene1,pairs.gene2)
    binary = mat_to_binary(v)
    binary_sorted = binary.sort_values(by=['gene1','gene2'])
    tuples_in_df = pd.MultiIndex.from_frame(binary_sorted[["gene1","gene2"]])
    fltr = binary_sorted[tuples_in_df.isin(pairstuple)]
    fltr_sorted = fltr.sort_values(by=['gene1', 'gene2'])
    new_arr.append(fltr_sorted.score.values)

result = pd.DataFrame(new_arr, columns=[i for i in range(5000)], index=arr_features).T
r = pd.concat([pairs.reset_index(drop=True),result],axis=1)

r['small_difference'] = r.apply(lambda x: x.SMALL_corr-x.SMALL_with_control_corr,axis=1)


#%%
# calculate means
df = pd.concat([df2,df1], axis=1)
df_mean = df.mean(axis=1)
df1_mean, df2_mean, df3_mean = df1.mean(axis=1), df2.mean(axis=1), df3.mean(axis=1)

r['gene1_mean'] = np.nan
r['gene2_mean'] = np.nan
r['gene1_X1_mean'] = np.nan
r['gene1_X2_mean'] = np.nan
r['gene1_X3_mean'] = np.nan
r['gene2_X1_mean'] = np.nan
r['gene2_X2_mean'] = np.nan
r['gene2_X3_mean'] = np.nan

for k,row in r.iterrows():
    gene1 = r.at[k,'gene1']
    gene2 = r.at[k,'gene2']
    r.at[k,'gene1_mean'] = df_mean.loc[gene1]
    r.at[k,'gene2_mean'] = df_mean.loc[gene2]
    r.at[k,'gene1_X1_mean'] = df1_mean.loc[gene1]
    r.at[k,'gene1_X2_mean'] = df2_mean.loc[gene1]
    r.at[k,'gene1_X3_mean'] = df3_mean.loc[gene1]
    r.at[k,'gene2_X1_mean'] = df1_mean.loc[gene2]
    r.at[k,'gene2_X2_mean'] = df2_mean.loc[gene2]
    r.at[k,'gene2_X3_mean'] = df3_mean.loc[gene2]

#%%
r['X2_pval'] = np.nan
r['X3_pval'] = np.nan
for k,row in r.iterrows():
    gene1 = r.at[k,'gene1']
    gene2 = r.at[k,'gene2']
    pcc_x2 = pearsonr(d2.loc[gene1], d2.loc[gene2])
    pcc_x3 = pearsonr(df3.loc[gene1], df3.loc[gene2])
    r.at[k,'X2_pval'] = pcc_x2[1]
    r.at[k,'X3_pval'] = pcc_x3[1]

#%%
from statsmodels.stats.multitest import multipletests
fdr_x2 = multipletests(r.X2_pval, alpha=0.05, method='fdr_bh')
fdr_x3 = multipletests(r.X3_pval, alpha=0.05, method='fdr_bh')
r['X2_fdr_corrected'] = fdr_x2[1]
r['X3_fdr_corrected'] = fdr_x3[1]
# %%
#### Scatter
##############################################################
from matplotlib.backends.backend_pdf import PdfPages
scatterdf = dfc.T.copy()
scatterdf['dataset'] = 0
scatterdf.iloc[:7,-1]=1

gp = np.split(pairs,25)
pdf = PdfPages("output.pdf")

sns.set_style('darkgrid')
for x in range(len(gp)):
    df = gp[x]
    fig, ax = plt.subplots(5, 4, figsize=(12, 15))
    axr = ax.ravel()
    for i in range(len(axr)):
            sns.scatterplot(data=scatterdf, x=df['gene1'].values[i], y=df['gene2'].values[i], hue='dataset', ax=axr[i], palette="viridis", legend=False)
    
    fig.tight_layout(pad=1)
    #plt.savefig(f'0000{x}.png')
    pdf.savefig( fig )
pdf.close()




# %%
### projection
###################################################
df1_projection = projection(df1s,df2s)
df2_projection = projection(df2s,df1s)

pr1_merged = pd.merge(df1,df2_projection,left_index=True,right_index=True)
pr2_merged = pd.merge(df2,df1_projection,left_index=True,right_index=True)

#pr1_merged.to_csv('X1_and_X2projection.csv')
#pr2_merged.to_csv('X2_and_X1projection.csv')

# sns.scatterplot(test, x='Met_x',y='Met_y', hue='hue', palette='viridis')
# plt.savefig(f'X1_dataset_comparing_projected_results2.png',dpi=300)
# %%
test = pr1_merged[['Met_x','Met_y']].copy()
test['hue'] = 0
test.loc[test.eval('Met_x>2'), 'hue'] = 1
test.loc[test.eval('Met_y>2'), 'hue'] = 2

sns.scatterplot(test, x='Met_x',y='Met_y', hue='hue', palette='viridis')

# %%
#%%

#x = dfs.loc['RPL15']
x = dfs.loc['LIPT2']

y = dfs.loc['SFXN1']
# x = gene1 vector
# y = gene2 vector
# column seperates dataset i.e: 7
column = 7
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
r = np.dot(xm,ym) / (normxm*normym)
r1 = np.dot(xAm,yAm) / (normxm*normym)
r2 = np.dot(xBm,yBm) / (normxm*normym)


# %%
