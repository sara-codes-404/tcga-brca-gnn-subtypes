import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif

# running locally now, everything from step 0 onward lives in data/
base = 'data/'

# load the data
df = pd.read_csv(base + 'tcga_brca.csv', index_col=0)
print('shape:', df.shape)

# last column is the label, the rest are genes
X = df.drop(columns=['subtype'])
y = df['subtype']
gene_names = list(X.columns)

# turn the subtype names into numbers 0-4
class_names = sorted(y.unique())
y = y.map({name: i for i, name in enumerate(class_names)})
print('classes:', class_names)

# split first, so test data doesnt leak into anything
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# remove low variance genes (only look at train)
min_var = 0.5
var = X_train.var()
keep = var[var > min_var].index
X_train = X_train[keep]
X_test = X_test[keep]
print('genes left:', len(keep), 'out of', len(gene_names))

# keep only the genes that separate the subtypes best (anova f-test, train only)
num_genes = 1000
F, p = f_classif(X_train, y_train)
F = np.nan_to_num(F)
best = np.argsort(F)[::-1][:num_genes]
keep = keep[best]
X_train = X_train[keep]
X_test = X_test[keep]
print('genes after f-test:', len(keep))

# normalize the data (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# save everything for the next step
np.save(base + 'X_train.npy', X_train)
np.save(base + 'X_test.npy', X_test)
np.save(base + 'y_train.npy', y_train.values)
np.save(base + 'y_test.npy', y_test.values)
pd.Series(list(keep)).to_csv(base + 'gene_names.csv', index=False)
pd.Series(class_names).to_csv(base + 'class_names.csv', index=False)

print('train:', X_train.shape, 'test:', X_test.shape)