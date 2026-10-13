import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

os.makedirs('results', exist_ok=True)

# same pipeline as step 9, but the question is different:
# is it the graph that matters? same network, 3 different edge sets
df = pd.read_csv('data/tcga_brca.csv', index_col=0)
X_all = df.drop(columns=['subtype']).values
class_names = sorted(df['subtype'].unique())
y_all = df['subtype'].map({n: i for i, n in enumerate(class_names)}).values
num_classes = len(class_names)
her2 = class_names.index('Her2')

class Net(torch.nn.Module):
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

def predict(model, x, edge_index):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    return out.argmax(dim=1).numpy()

def train_net(x, edge_index, y, train_part, val_part, seed):
    torch.manual_seed(seed)
    model = Net(x.shape[1], 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    y_t = torch.tensor(y, dtype=torch.long)
    best_val = -1
    best_state = None
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_part], y_t[train_part])
        loss.backward()
        optimizer.step()

        pred = predict(model, x, edge_index)
        val_f1 = f1_score(y[val_part], pred[val_part], average='macro', zero_division=0)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(model, x, edge_index)

def score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_each = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    return [acc, f1, f1_each[her2]]

names = ['LogReg', 'MLP (no edges)', 'GCN (random edges)', 'GCN (knn edges)']
results = {n: [] for n in names}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
start = time.time()

for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
    X_tr = X_all[train_idx]
    X_te = X_all[test_idx]
    y_tr = y_all[train_idx]
    y_te = y_all[test_idx]

    # remove low variance genes, f-test top 1000, normalize (train fold only)
    keep = np.where(X_tr.var(axis=0, ddof=1) > 0.5)[0]
    X_tr = X_tr[:, keep]
    X_te = X_te[:, keep]
    f_vals, p_vals = f_classif(X_tr, y_tr)
    f_vals = np.nan_to_num(f_vals)
    best = np.argsort(f_vals)[::-1][:1000]
    X_tr = X_tr[:, best]
    X_te = X_te[:, best]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    num_train = len(X_tr)
    num_nodes = len(X)
    test_part = np.arange(num_train, num_nodes)
    train_part, val_part = train_test_split(
        np.arange(num_train), test_size=0.15, random_state=fold, stratify=y_tr)
    x = torch.tensor(X, dtype=torch.float)

    # the real knn graph
    A = kneighbors_graph(X, n_neighbors=10, mode='connectivity')
    rows, cols = A.nonzero()
    knn_edges = to_undirected(torch.tensor(np.array([rows, cols]), dtype=torch.long))

    # random graph: same edges, but the patient ids are shuffled.
    # every patient keeps the same number of neighbors, but they are random patients
    torch.manual_seed(fold)
    perm = torch.randperm(num_nodes)
    random_edges = perm[knn_edges]

    # no edges at all. GCNConv only adds self loops, so this is a normal MLP
    no_edges = torch.empty((2, 0), dtype=torch.long)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X[train_part], y[train_part])
    results['LogReg'].append(score(y[test_part], lr.predict(X[test_part])))

    for name, edges in [('MLP (no edges)', no_edges),
                        ('GCN (random edges)', random_edges),
                        ('GCN (knn edges)', knn_edges)]:
        pred = train_net(x, edges, y, train_part, val_part, fold)
        results[name].append(score(y[test_part], pred[test_part]))

    print('fold', fold, 'done,', round(time.time() - start), 'sec so far')
    for name in names:
        r = results[name][-1]
        print('   ', name, 'acc', round(r[0], 3), 'macro f1', round(r[1], 3), 'her2 f1', round(r[2], 3))

def ms(a):
    return str(round(a.mean(), 3)) + ' +- ' + str(round(a.std(), 3))

rows = []
for name in names:
    s = np.array(results[name])
    rows.append({'model': name, 'test_acc': ms(s[:, 0]),
                 'macro_f1': ms(s[:, 1]), 'her2_f1': ms(s[:, 2])})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv('results/graph_controls.csv', index=False)

# same folds for every model, so look at the difference fold by fold
knn = np.array(results['GCN (knn edges)'])[:, 1]
for name in ['MLP (no edges)', 'GCN (random edges)', 'LogReg']:
    d = knn - np.array(results[name])[:, 1]
    print('knn GCN minus', name, '(macro f1 per fold):', np.round(d, 3), '| mean', round(d.mean(), 3))
print('saved to results/graph_controls.csv')