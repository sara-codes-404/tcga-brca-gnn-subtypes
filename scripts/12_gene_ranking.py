import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

os.makedirs('results/plots', exist_ok=True)

top_n = 20   # how many genes we look at for each subtype

# raw data
df = pd.read_csv('data/tcga_brca.csv', index_col=0)
X_all = df.drop(columns=['subtype']).values
gene_names = np.array(df.drop(columns=['subtype']).columns)
class_names = sorted(df['subtype'].unique())
y_all = df['subtype'].map({n: i for i, n in enumerate(class_names)}).values
num_classes = len(class_names)

# pam50 genes (official list from Parker 2009, ORC6 is the new name of ORC6L)
pam50 = ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1',
         'CDC20', 'CDC6', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2',
         'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C',
         'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67',
         'MLPH', 'MMP11', 'MYBL2', 'MYC', 'NAT1', 'NDC80', 'NUF2', 'ORC6L', 'ORC6',
         'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B',
         'TYMS', 'UBE2C', 'UBE2T']

# welch t-test, one subtype against all the others (no model, just statistics)
# positive = gene is higher in this subtype, negative = lower
def welch_t(X, y, c):
    a = X[y == c]
    b = X[y != c]
    diff = a.mean(axis=0) - b.mean(axis=0)
    se = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
    return diff / (se + 1e-8)

# t[c] = list with one Series per fold (index = gene name)
t_scores = [[] for c in range(num_classes)]
num_kept = []
pam_kept = []   # pam50 genes left after the variance filter

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
    # only the train fold is used, same folds as steps 9 and 11
    X_tr = X_all[train_idx]
    y_tr = y_all[train_idx]
    keep = np.where(X_tr.var(axis=0, ddof=1) > 0.5)[0]   # remove low variance genes
    names = gene_names[keep]
    num_kept.append(len(keep))
    pam_kept.append(int(np.isin(names, pam50).sum()))
    for c in range(num_classes):
        t_scores[c].append(pd.Series(welch_t(X_tr[:, keep], y_tr, c), index=names))
    print('fold', fold, 'done')

def jaccard(a, b):
    return len(a & b) / len(a | b)

rows = []
pam_counts = []
stability = []
for c in range(num_classes):
    # top genes by size of t (up or down), for every fold
    tops = [set(s.abs().nlargest(top_n).index) for s in t_scores[c]]
    for f in range(len(tops)):
        pam_counts.append(len(tops[f] & set(pam50)))
    for a, b in itertools.combinations(range(len(tops)), 2):
        stability.append(jaccard(tops[a], tops[b]))
    table = pd.concat(t_scores[c], axis=1)
    counts = pd.Series(list(itertools.chain(*[list(t) for t in tops]))).value_counts()
    for gene, n in counts.items():
        mean_t = float(table.loc[gene].mean())
        # does the sign stay the same in all folds?
        same_sign = bool((np.sign(table.loc[gene].dropna()) == np.sign(mean_t)).all())
        rows.append([class_names[c], gene, 'up' if mean_t > 0 else 'down',
                     int(n), round(mean_t, 2), same_sign, gene in pam50])

res = pd.DataFrame(rows, columns=['subtype', 'gene', 'direction', 'folds_in_top',
                                  'mean_t', 'same_sign_in_all_folds', 'is_pam50'])
res['abs_t'] = res['mean_t'].abs()
res = res.sort_values(['subtype', 'folds_in_top', 'abs_t'], ascending=[True, False, False])
res = res.drop(columns=['abs_t'])
res.to_csv('results/stable_genes_ttest.csv', index=False)

print()
print('genes in the top', top_n, 'of at least 4 folds (up/down compared to the other subtypes):')
for c in range(num_classes):
    r = res[(res['subtype'] == class_names[c]) & (res['folds_in_top'] >= 4)]
    items = []
    for g, d, p in zip(r['gene'], r['direction'], r['is_pam50']):
        items.append(g + ('*' if p else '') + ('+' if d == 'up' else '-'))
    print(class_names[c], '(' + str(len(items)) + ' genes):', ', '.join(items))
print('(* = pam50 gene, + = higher in this subtype, - = lower)')

expected = top_n * np.mean(pam_kept) / np.mean(num_kept)
print()
print('pam50 genes in the top', top_n, '(average per subtype and fold):', round(np.mean(pam_counts), 2),
      '| by luck:', round(expected, 2))
print('similarity of top lists between folds (jaccard, 1 = always same genes):', round(np.mean(stability), 2))

# plot: best stable genes of every subtype
fig, axes = plt.subplots(1, num_classes, figsize=(16, 4.5))
for c in range(num_classes):
    r = res[(res['subtype'] == class_names[c]) & (res['folds_in_top'] >= 4)].copy()
    r['abs_t'] = r['mean_t'].abs()
    r = r.sort_values('abs_t', ascending=False).head(10)
    labels = [g + ('*' if p else '') for g, p in zip(r['gene'], r['is_pam50'])]
    axes[c].barh(labels[::-1], r['mean_t'].values[::-1])
    axes[c].axvline(0, color='black', linewidth=0.8)
    axes[c].set_title(class_names[c])
    axes[c].set_xlabel('mean t (+ higher in subtype)')
plt.tight_layout()
plt.savefig('results/plots/stable_genes_ttest.png')
print('saved results/stable_genes_ttest.csv and results/plots/stable_genes_ttest.png')