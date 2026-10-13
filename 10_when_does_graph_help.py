import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

os.makedirs('results/plots', exist_ok=True)

# settings (change these if its too slow)
fractions = [0.05, 0.1, 0.2, 0.5, 0.8]   # how much of the data has labels
num_repeats = 3                          # random splits for each fraction
num_genes = 1000
k = 10
epochs = 100
do_ablation = True                       # part 2
do_k_sweep = True                        # part 3

# raw data, no filtering yet
df = pd.read_csv('data/tcga_brca.csv', index_col=0)
X_all = df.drop(columns=['subtype']).values
gene_names = np.array(df.drop(columns=['subtype']).columns)
class_names = sorted(df['subtype'].unique())
y_all = df['subtype'].map({n: i for i, n in enumerate(class_names)}).values
num_classes = len(class_names)
num_nodes = len(y_all)
her2 = class_names.index('Her2')

# pam50 genes (from memory, same list as step 5)
pam50 = ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1',
         'CDC20', 'CDC6', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2',
         'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C',
         'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67',
         'MLPH', 'MMP11', 'MYBL2', 'NAT1', 'NDC80', 'NUF2', 'ORC6L', 'ORC6',
         'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B',
         'TYMS', 'UBE2C', 'UBE2T', 'KNTC2', 'CDCA1']

class Net(torch.nn.Module):
    def __init__(self, conv, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = conv(in_dim, hidden_dim)
        self.conv2 = conv(hidden_dim, out_dim)
        self.dropout = 0.5

    def forward(self, x, edge_index, x_nei=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# mean of the neighbors features. it never changes while training,
# so we do it once (this is what makes SAGE slow with 1000 features otherwise)
def neighbor_mean(x, edge_index):
    src, dst = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    deg = torch.zeros(x.shape[0])
    deg.index_add_(0, dst, torch.ones(src.shape[0]))
    return out / deg.clamp(min=1).view(-1, 1)

# 2 layer GraphSAGE (mean aggregation). first layer is written by hand,
# it computes the same thing as the first SAGEConv layer, just faster
class SageNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin_self = torch.nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin_nei = torch.nn.Linear(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = 0.5

    def forward(self, x, edge_index, x_nei):
        h = F.relu(self.lin_self(x) + self.lin_nei(x_nei))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)

# fixed number of epochs, no val set (with 5% labels a val set is useless)
def train_net(kind, x, edge_index, y_t, labeled, seed):
    torch.manual_seed(seed)
    x_nei = None
    if kind == 'SAGE':
        model = SageNet(x.shape[1], 64, num_classes)
        x_nei = neighbor_mean(x, edge_index)
    else:
        model = Net(GCNConv, x.shape[1], 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    lab = torch.tensor(labeled, dtype=torch.long)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, x_nei)
        loss = F.cross_entropy(out[lab], y_t[lab])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, x_nei)
    return out.argmax(dim=1).numpy()

def score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_each = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    return acc, f1, f1_each[her2]

# one experiment: some patients have labels, all the others are the test set
def run_one(X_raw, labeled, test, seed, k_nn, model_names):
    # no labels used in these 2 steps (graph models see all the patients anyway)
    keep = np.where(X_raw.var(axis=0, ddof=1) > 0.5)[0]
    X = X_raw[:, keep]
    # f-test only looks at the labeled patients
    f_vals, p_vals = f_classif(X[labeled], y_all[labeled])
    f_vals = np.nan_to_num(f_vals)
    best = np.argsort(f_vals)[::-1][:num_genes]
    X = X[:, best]
    X = StandardScaler().fit_transform(X)

    out = {}
    if 'LogReg' in model_names:
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X[labeled], y_all[labeled])
        out['LogReg'] = score(y_all[test], lr.predict(X[test]))

    graph_models = [m for m in model_names if m != 'LogReg']
    if len(graph_models) > 0:
        A = kneighbors_graph(X, n_neighbors=k_nn, mode='connectivity')
        rows, cols = A.nonzero()
        edge_index = to_undirected(torch.tensor(np.array([rows, cols]), dtype=torch.long))
        x = torch.tensor(X, dtype=torch.float)
        y_t = torch.tensor(y_all, dtype=torch.long)
        for m in graph_models:
            pred = train_net(m, x, edge_index, y_t, labeled, seed)
            out[m] = score(y_all[test], pred[test])
    return out

def get_split(fraction, rep):
    labeled, test = train_test_split(
        np.arange(num_nodes), train_size=fraction, stratify=y_all, random_state=rep)
    return labeled, test

all_models = ['LogReg', 'GCN', 'SAGE']
results = []   # one row per (part, setting, model, repeat)
start = time.time()

# part 1: how does it change when we have fewer labels?
print('part 1: fewer and fewer labels')
for fraction in fractions:
    for rep in range(num_repeats):
        labeled, test = get_split(fraction, rep)
        out = run_one(X_all, labeled, test, rep, k, all_models)
        for m in out:
            results.append(['labels', fraction, m, rep] + list(out[m]))
        print('  fraction', fraction, 'repeat', rep, 'done,', round(time.time() - start), 'sec')

# part 2: take the 50 pam50 genes away. the labels come from these genes,
# so now the model has to find the subtypes from the other genes
if do_ablation:
    print('part 2: without the pam50 genes')
    keep_cols = ~np.isin(gene_names, pam50)
    print('  removed', int((~keep_cols).sum()), 'pam50 genes')
    X_no_pam = X_all[:, keep_cols]
    for fraction in [0.1, 0.8]:
        for rep in range(num_repeats):
            labeled, test = get_split(fraction, rep)
            out = run_one(X_no_pam, labeled, test, rep, k, all_models)
            for m in out:
                results.append(['no_pam50', fraction, m, rep] + list(out[m]))
            print('  fraction', fraction, 'repeat', rep, 'done,', round(time.time() - start), 'sec')

# part 3: different k for the graph (LogReg doesnt use the graph so its left out)
if do_k_sweep:
    print('part 3: different k')
    for k_nn in [3, 5, 10, 20, 40]:
        for fraction in [0.1, 0.8]:
            for rep in range(num_repeats):
                labeled, test = get_split(fraction, rep)
                out = run_one(X_all, labeled, test, rep, k_nn, ['GCN', 'SAGE'])
                for m in out:
                    results.append(['k=' + str(k_nn), fraction, m, rep] + list(out[m]))
            print('  k', k_nn, 'fraction', fraction, 'done,', round(time.time() - start), 'sec')

res = pd.DataFrame(results, columns=['part', 'fraction', 'model', 'repeat', 'acc', 'macro_f1', 'her2_f1'])
res.to_csv('results/graph_help_all_runs.csv', index=False)

def show(part, metric):
    t = res[res['part'] == part]
    mean = t.groupby(['model', 'fraction'])[metric].mean().unstack()
    std = t.groupby(['model', 'fraction'])[metric].std().unstack()
    out = mean.round(3).astype(str) + ' +- ' + std.round(3).astype(str)
    return out, mean

print()
print('===== part 1: macro f1, columns = fraction of patients with labels =====')
table1, mean1 = show('labels', 'macro_f1')
print(table1.to_string())
print()
print('accuracy:')
print(show('labels', 'acc')[0].to_string())
print()
print('difference to LogReg (macro f1, positive = graph model is better):')
for m in ['GCN', 'SAGE']:
    print(m, (mean1.loc[m] - mean1.loc['LogReg']).round(3).to_dict())

if do_ablation:
    print()
    print('===== part 2: without pam50 genes (macro f1) =====')
    print(show('no_pam50', 'macro_f1')[0].to_string())

if do_k_sweep:
    print()
    print('===== part 3: k sweep (macro f1) =====')
    ks = res[res['part'].str.startswith('k=')]
    kt = ks.groupby(['part', 'model', 'fraction'])['macro_f1'].mean().unstack().round(3)
    print(kt.to_string())

# plot for part 1
plt.figure(figsize=(6, 4))
t = res[res['part'] == 'labels']
for m in all_models:
    g = t[t['model'] == m].groupby('fraction')['macro_f1']
    plt.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, marker='o', capsize=3, label=m)
plt.xlabel('fraction of patients with labels')
plt.ylabel('macro F1 on the other patients')
plt.legend()
plt.tight_layout()
plt.savefig('results/plots/label_fraction.png')
print()
print('saved results/graph_help_all_runs.csv and results/plots/label_fraction.png')