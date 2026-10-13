import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

os.makedirs('results', exist_ok=True)

top_n = 20          # how many genes we look at for each subtype
num_genes = 1000
epochs = 100

# raw data
df = pd.read_csv('data/tcga_brca.csv', index_col=0)
X_all = df.drop(columns=['subtype']).values
gene_names = np.array(df.drop(columns=['subtype']).columns)
class_names = sorted(df['subtype'].unique())
y_all = df['subtype'].map({n: i for i, n in enumerate(class_names)}).values
num_classes = len(class_names)

# pam50 genes (from memory, same list as step 5)
pam50 = ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1',
         'CDC20', 'CDC6', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2',
         'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C',
         'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67',
         'MLPH', 'MMP11', 'MYBL2', 'NAT1', 'NDC80', 'NUF2', 'ORC6L', 'ORC6',
         'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B',
         'TYMS', 'UBE2C', 'UBE2T', 'KNTC2', 'CDCA1']

# same 2-layer GCN as step 3
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = 0.5

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# importance[method][class] = list with one Series per fold (index = gene name)
importance = {'LogReg': [[] for c in range(num_classes)],
              'GCN': [[] for c in range(num_classes)]}
pam_in_data = []   # how many pam50 genes are in the 1000 genes, per fold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
    X_tr = X_all[train_idx]
    X_te = X_all[test_idx]
    y_tr = y_all[train_idx]
    y_te = y_all[test_idx]

    # everything is done on the train fold only (same as step 9)
    keep = np.where(X_tr.var(axis=0, ddof=1) > 0.5)[0]
    f_vals, p_vals = f_classif(X_tr[:, keep], y_tr)
    f_vals = np.nan_to_num(f_vals)
    best = keep[np.argsort(f_vals)[::-1][:num_genes]]
    names = gene_names[best]
    pam_in_data.append(int(np.isin(names, pam50).sum()))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr[:, best])
    X_te = scaler.transform(X_te[:, best])

    # method 1: logistic regression. one row of coefficients for each subtype
    # (big positive coefficient = gene pushes the patient towards that subtype)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_tr, y_tr)
    for c in range(num_classes):
        importance['LogReg'][c].append(pd.Series(lr.coef_[c], index=names))

    # method 2: gradient of the GCN. how much does the logit of subtype c go up
    # when a gene goes up, for the train patients of subtype c?
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    num_train = len(X_tr)
    A = kneighbors_graph(X, n_neighbors=10, mode='connectivity')
    rows, cols = A.nonzero()
    edge_index = to_undirected(torch.tensor(np.array([rows, cols]), dtype=torch.long))
    x = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)

    torch.manual_seed(fold)
    model = GCN(x.shape[1], 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_nodes = torch.arange(num_train)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_nodes], y_t[train_nodes])
        loss.backward()
        optimizer.step()

    model.eval()
    for c in range(num_classes):
        xg = x.clone()
        xg.requires_grad_(True)
        out = model(xg, edge_index)
        nodes_c = torch.where(y_t[:num_train] == c)[0]
        out[nodes_c, c].sum().backward()
        g = xg.grad.sum(dim=0) / len(nodes_c)
        importance['GCN'][c].append(pd.Series(g.numpy(), index=names))
    print('fold', fold, 'done')

def jaccard(a, b):
    return len(a & b) / len(a | b)

summary = []
for method in importance:
    all_rows = []
    pam_counts = []
    stability = []
    for c in range(num_classes):
        tops = [set(s.nlargest(top_n).index) for s in importance[method][c]]
        # how many pam50 genes in the top list of each fold
        for f in range(len(tops)):
            pam_counts.append(len(tops[f] & set(pam50)))
        # how similar are the top lists of different folds
        for a, b in itertools.combinations(range(len(tops)), 2):
            stability.append(jaccard(tops[a], tops[b]))
        # genes that show up in the top list of many folds
        table = pd.concat(importance[method][c], axis=1)
        counts = pd.Series(list(itertools.chain(*[list(t) for t in tops]))).value_counts()
        for gene, n in counts.items():
            all_rows.append([class_names[c], gene, int(n), gene in pam50,
                             round(float(table.loc[gene].mean()), 4)])
    res = pd.DataFrame(all_rows, columns=['subtype', 'gene', 'folds_in_top', 'is_pam50', 'mean_importance'])
    res = res.sort_values(['subtype', 'folds_in_top', 'mean_importance'], ascending=[True, False, False])
    res.to_csv('results/stable_genes_' + method + '.csv', index=False)

    print()
    print('=====', method, '=====')
    for c in range(num_classes):
        r = res[(res['subtype'] == class_names[c]) & (res['folds_in_top'] >= 4)]
        names_list = [g + ('*' if p else '') for g, p in zip(r['gene'], r['is_pam50'])]
        print(class_names[c], '(in top', top_n, 'of at least 4 folds):', ', '.join(names_list))
    expected = top_n * np.mean(pam_in_data) / num_genes
    print('(* = pam50 gene)')
    print('pam50 genes in the top', top_n, '(average per subtype and fold):',
          round(np.mean(pam_counts), 2), '| by luck:', round(expected, 2))
    print('similarity of top lists between folds (jaccard, 1 = always same genes):',
          round(np.mean(stability), 2))
    summary.append([method, round(np.mean(pam_counts), 2), round(expected, 2), round(np.mean(stability), 2)])

# do the two methods find the same genes?
overlap = []
for c in range(num_classes):
    for f in range(5):
        a = set(importance['LogReg'][c][f].nlargest(top_n).index)
        b = set(importance['GCN'][c][f].nlargest(top_n).index)
        overlap.append(len(a & b))
print()
print('genes shared by LogReg and GCN in the top', top_n, '(average):', round(np.mean(overlap), 1))
pd.DataFrame(summary, columns=['method', 'pam50_in_top', 'pam50_by_luck', 'stability']).to_csv(
    'results/stable_genes_summary.csv', index=False)
print('saved results/stable_genes_LogReg.csv, results/stable_genes_GCN.csv and results/stable_genes_summary.csv')