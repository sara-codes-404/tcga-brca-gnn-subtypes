import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

base = 'data/'
os.makedirs('results', exist_ok=True)

data = torch.load(base + 'graph.pt', weights_only=False)
class_names = pd.read_csv(base + 'class_names.csv').iloc[:, 0].tolist()
num_classes = len(class_names)
her2 = class_names.index('Her2')

X = data.x.numpy()
y = data.y.numpy()
train = data.train_mask.numpy()
val = data.val_mask.numpy()
test = data.test_mask.numpy()

# part 1: for each subtype, how many of its neighbors have the same subtype?
src = data.edge_index[0].numpy()
dst = data.edge_index[1].numpy()
same = (y[src] == y[dst])
for c in range(num_classes):
    m = (y[src] == c)
    print(class_names[c], '| nodes:', int((y == c).sum()),
          '| same-subtype neighbors:', round(same[m].mean(), 3))

# part 2: same 2-layer net, but we can swap the conv layer
class Net(torch.nn.Module):
    def __init__(self, conv, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = conv(in_dim, hidden_dim)
        self.conv2 = conv(hidden_dim, out_dim)
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

# rare classes get a bigger weight in the loss
counts = np.bincount(y[train], minlength=num_classes)
class_w = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float)

# best epoch is picked by val macro f1 (same rule for every variant)
def train_net(conv, use_weight, seed):
    torch.manual_seed(seed)
    model = Net(conv, data.x.shape[1], 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    w = class_w if use_weight else None
    best_val = -1
    best_state = None
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=w)
        loss.backward()
        optimizer.step()

        pred = predict(model)
        val_f1 = f1_score(y[val], pred[val], average='macro', zero_division=0)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(model), best_val

# accuracy, macro f1 and her2 f1 on test nodes
def score(pred):
    acc = accuracy_score(y[test], pred[test])
    f1 = f1_score(y[test], pred[test], average='macro')
    f1_each = f1_score(y[test], pred[test], average=None, labels=list(range(num_classes)))
    return acc, f1, f1_each[her2]

variants = {
    'GCN': (GCNConv, False),
    'GCN + class weights': (GCNConv, True),
    'SAGE': (SAGEConv, False),
    'SAGE + class weights': (SAGEConv, True)}

results = {}
for name in variants:
    conv, use_weight = variants[name]
    rows = []
    start = time.time()
    for seed in range(5):
        pred, val_f1 = train_net(conv, use_weight, seed)
        rows.append(score(pred) + (val_f1,))
        print('  ', name, 'seed', seed, 'finished,', round(time.time() - start), 'sec so far')
    results[name] = np.array(rows)
    print('done:', name)

# logistic regression again, just as a reference line
lr = LogisticRegression(max_iter=2000)
lr.fit(X[train], y[train])
lr_pred = lr.predict(X)
lr_val = f1_score(y[val], lr_pred[val], average='macro', zero_division=0)
results['LogReg (reference)'] = np.array([score(lr_pred) + (lr_val,)])

def ms(a):
    return str(round(a.mean(), 3)) + ' +- ' + str(round(a.std(), 3))

rows = []
for name in results:
    s = results[name]
    rows.append({'model': name, 'val_f1': ms(s[:, 3]), 'test_acc': ms(s[:, 0]),
                 'macro_f1': ms(s[:, 1]), 'her2_f1': ms(s[:, 2])})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv('results/gcn_variants.csv', index=False)
print('saved to results/gcn_variants.csv')