import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv

base = 'data/'
os.makedirs('results/plots', exist_ok=True)

gene_names = pd.read_csv(base + 'gene_names.csv').iloc[:, 0].tolist()
data = torch.load(base + 'graph.pt', weights_only=False)
num_classes = int(data.y.max()) + 1

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

state = torch.load(base + 'best_model.pt', map_location='cpu')
model = GCN(data.x.shape[1], 64, num_classes)
model.load_state_dict(state)
model.eval()

# PAM50 genes (typed from memory, a few old names too). the subtype labels
# come from these genes so a good method should find some of them
pam50 = ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1',
         'CDC20', 'CDC6', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2',
         'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C',
         'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67',
         'MLPH', 'MMP11', 'MYBL2', 'NAT1', 'NDC80', 'NUF2', 'ORC6L', 'ORC6',
         'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B',
         'TYMS', 'UBE2C', 'UBE2T', 'KNTC2', 'CDCA1']

# method 1: get weights from first layer
if 'conv1.lin.weight' in state:
    w = state['conv1.lin.weight'].numpy()   # hidden x genes
else:
    w = state['conv1.weight'].numpy().T
imp_w = np.abs(w).mean(axis=0)

# method 2: gradient. how much does the output move if a gene changes?
x = data.x.clone()
x.requires_grad_(True)
out = model(x, data.edge_index)
pred = out.argmax(dim=1).detach()
score = out.gather(1, pred.view(-1, 1)).sum()   # logit of predicted class
score.backward()
imp_g = x.grad.abs().mean(dim=0).numpy()

def make_table(imp):
    t = pd.DataFrame({'gene': gene_names, 'importance': imp})
    t = t.sort_values('importance', ascending=False)
    t['is_pam50'] = t['gene'].isin(pam50)
    return t

def report(name, t):
    print('----', name, '----')
    print('max / median importance:', round(t['importance'].max(), 5), '/', round(t['importance'].median(), 5))
    print(t.head(20).to_string(index=False))
    # how many pam50 genes in the top 100, compared to pure luck
    num_pam = int(t['gene'].isin(pam50).sum())
    in_top = int(t.head(100)['is_pam50'].sum())
    luck = 100 * num_pam / len(t)
    print('pam50 genes in data:', num_pam)
    print('pam50 genes in top 100:', in_top, '(by luck we would expect about', round(luck, 1), ')')

table_w = make_table(imp_w)
table_g = make_table(imp_g)
report('weights', table_w)
report('gradient', table_g)

# save
table_w.head(20).to_csv('results/important_genes_weights.csv', index=False)
table_g.head(20).to_csv('results/important_genes.csv', index=False)

# plot top 20 genes (gradient method)
top = table_g.head(20)
plt.figure(figsize=(7, 6))
plt.barh(top['gene'][::-1], top['importance'][::-1])
plt.xlabel('mean abs gradient')
plt.title('top 20 genes (gradient)')
plt.tight_layout()
plt.savefig('results/plots/top_genes.png')
print('saved results/important_genes.csv and results/plots/top_genes.png')