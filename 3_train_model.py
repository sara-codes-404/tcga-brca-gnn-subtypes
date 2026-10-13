import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# running locally now, everything from step 0 onward lives in data/
base = 'data/'

# load the graph from step 2
data = torch.load(base + 'graph.pt', weights_only=False)
num_classes = len(data.y.unique())
num_features = data.x.shape[1]
print('features:', num_features, 'classes:', num_classes)

# 2-layer GCN
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using:', device)

model = GCN(in_dim=num_features, hidden_dim=64, out_dim=num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc

# training loop
best_val_acc = 0
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    val_acc = evaluate(data.val_mask)
    # save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), base + 'best_model.pt')

    if epoch % 20 == 0:
        train_acc = evaluate(data.train_mask)
        print(f'epoch {epoch}, loss {loss.item():.3f}, train acc {train_acc:.3f}, val acc {val_acc:.3f}')

print('best val acc:', best_val_acc)
print('saved to ' + base + 'best_model.pt')