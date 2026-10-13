import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# running locally now, everything from step 0 onward lives in data/
base = 'data/'

# load the stuff from step 1
X_train = np.load(base + 'X_train.npy')
X_test = np.load(base + 'X_test.npy')
y_train = np.load(base + 'y_train.npy')
y_test = np.load(base + 'y_test.npy')

# put train and test together, each patient is one node
X = np.vstack([X_train, X_test])
y = np.concatenate([y_train, y_test])
num_train = len(X_train)
num_nodes = len(X)

# build graph using k-nearest neighbors
k = 10
A = kneighbors_graph(X, n_neighbors=k, mode='connectivity')
rows, cols = A.nonzero()
edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
edge_index = to_undirected(edge_index)  # so edges go both ways
print('edges:', edge_index.shape[1])

# split train into train and val (val is for picking the best model later)
idx = np.arange(num_train)
train_idx, val_idx = train_test_split(
    idx, test_size=0.15, random_state=42, stratify=y_train)
test_idx = np.arange(num_train, num_nodes)

def make_mask(ids):
    m = torch.zeros(num_nodes, dtype=torch.bool)
    m[ids] = True
    return m

train_mask = make_mask(train_idx)
val_mask = make_mask(val_idx)
test_mask = make_mask(test_idx)

# convert to PyTorch Geometric format
data = Data(
    x=torch.tensor(X, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y, dtype=torch.long))
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print(data)
print('train/val/test:', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

# how many neighbors have the same subtype? quick check that the graph makes sense
src, dst = edge_index
same = (data.y[src] == data.y[dst]).float().mean().item()
print('edges connecting same subtype:', round(same, 3))

torch.save(data, base + 'graph.pt')
print('saved to ' + base + 'graph.pt')