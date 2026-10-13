import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

base = 'data/'
os.makedirs('results/plots', exist_ok=True)

# load the graph and the class names
data = torch.load(base + 'graph.pt', weights_only=False)
class_names = pd.read_csv(base + 'class_names.csv').iloc[:, 0].tolist()
num_classes = len(class_names)
num_nodes = data.num_nodes

# same GCN as step 3 (has to match or the weights wont load)
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

# load model
model = GCN(data.x.shape[1], 64, num_classes)
model.load_state_dict(torch.load(base + 'best_model.pt', map_location='cpu'))
model.eval()

# predict, no training here
with torch.no_grad():
    out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)

# only look at the test nodes
mask = data.test_mask
y_true = data.y[mask].numpy()
y_pred = pred[mask].numpy()

# calculate metrics
acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
print('test accuracy:', round(acc, 3))
print('macro F1:', round(f1_macro, 3))
print('weighted F1:', round(f1_weighted, 3))
print(classification_report(y_true, y_pred, target_names=class_names))

# plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(num_classes), class_names, rotation=45)
plt.yticks(range(num_classes), class_names)
for i in range(num_classes):
    for j in range(num_classes):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        plt.text(j, i, cm[i, j], ha='center', va='center', color=color)
plt.xlabel('predicted')
plt.ylabel('true')
plt.title('confusion matrix (test)')
plt.tight_layout()
plt.savefig('results/plots/confusion_matrix.png')
print('saved plot to results/plots/confusion_matrix.png')