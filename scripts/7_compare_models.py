import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

base = 'data/'
os.makedirs('results', exist_ok=True)

data = torch.load(base + 'graph.pt', weights_only=False)
class_names = pd.read_csv(base + 'class_names.csv').iloc[:, 0].tolist()
num_classes = len(class_names)
her2 = class_names.index('Her2')

# same train/val/test nodes for every model, so the comparison is fair
X = data.x.numpy()
y = data.y.numpy()
train = data.train_mask.numpy()
val = data.val_mask.numpy()
test = data.test_mask.numpy()

# same GCN as step 3
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

def predict(model):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return out.argmax(dim=1).numpy()

# train the gcn with a fixed seed, keep the epoch with best val acc
def train_gcn(seed):
    torch.manual_seed(seed)
    model = GCN(data.x.shape[1], 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val = 0
    best_state = None
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        pred = predict(model)
        val_acc = accuracy_score(y[val], pred[val])
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(model)

# accuracy, macro f1 and f1 of the her2 class, on test nodes only
def score(pred):
    acc = accuracy_score(y[test], pred[test])
    f1 = f1_score(y[test], pred[test], average='macro')
    f1_each = f1_score(y[test], pred[test], average=None, labels=list(range(num_classes)))
    return acc, f1, f1_each[her2]

results = {'GCN': [], 'LogReg': [], 'RandomForest': []}
for seed in range(5):
    print('seed', seed)
    results['GCN'].append(score(train_gcn(seed)))

    # baselines see the same train nodes but no graph
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X[train], y[train])
    results['LogReg'].append(score(lr.predict(X)))

    rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    rf.fit(X[train], y[train])
    results['RandomForest'].append(score(rf.predict(X)))

# mean +- std over the seeds
rows = []
for name in results:
    s = np.array(results[name])
    rows.append({
        'model': name,
        'test_acc': str(round(s[:, 0].mean(), 3)) + ' +- ' + str(round(s[:, 0].std(), 3)),
        'macro_f1': str(round(s[:, 1].mean(), 3)) + ' +- ' + str(round(s[:, 1].std(), 3)),
        'her2_f1': str(round(s[:, 2].mean(), 3)) + ' +- ' + str(round(s[:, 2].std(), 3))})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv('results/model_comparison.csv', index=False)
print('saved to results/model_comparison.csv')