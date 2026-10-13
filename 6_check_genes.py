import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

base = 'data/'
gene_names = pd.read_csv(base + 'gene_names.csv').iloc[:, 0].tolist()
X_train = np.load(base + 'X_train.npy')
y_train = np.load(base + 'y_train.npy')

# same pam50 list as step 5 (from memory, a few old names too)
pam50 = ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1',
         'CDC20', 'CDC6', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2',
         'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C',
         'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67',
         'MLPH', 'MMP11', 'MYBL2', 'NAT1', 'NDC80', 'NUF2', 'ORC6L', 'ORC6',
         'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B',
         'TYMS', 'UBE2C', 'UBE2T', 'KNTC2', 'CDCA1']

# part 1: anova f-test, no model at all. which genes separate the 5 subtypes?
F, p = f_classif(X_train, y_train)
F = np.nan_to_num(F)
t = pd.DataFrame({'gene': gene_names, 'F': F})
t = t.sort_values('F', ascending=False)
t['is_pam50'] = t['gene'].isin(pam50)
print(t.head(20).to_string(index=False))

num_pam = int(t['gene'].isin(pam50).sum())
in_top = int(t.head(100)['is_pam50'].sum())
luck = 100 * num_pam / len(t)
print('pam50 genes in top 100:', in_top, '(by luck about', round(luck, 1), ')')
t.head(20).to_csv('results/important_genes_ftest.csv', index=False)

# part 2: look at the raw expression of the top gradient genes from step 5
top_g = pd.read_csv('results/important_genes.csv')['gene'].tolist()
raw = pd.read_csv(base + 'tcga_brca.csv', index_col=0)
raw = raw.drop(columns=['subtype'])
stats = pd.DataFrame({
    'mean': raw[top_g].mean(),
    'pct_below_1': (raw[top_g] < 1).mean() * 100})
print(stats.round(2))
print('all genes, median of means:', round(raw.mean().median(), 2))
print('all genes, median pct below 1:', round((raw < 1).mean().median() * 100, 2))