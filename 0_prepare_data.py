import pandas as pd

# running locally now, raw files sit next to this script, outputs go in data/
in_dir = ""
out_dir = "data/"

# path to the two files we downloaded from xena
expr_path = in_dir + "HiSeqV2.gz"
clin_path = in_dir + "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"

# expression file: rows = genes, cols = samples. gotta transpose it
print("loading expression data...")
expr = pd.read_csv(expr_path, sep="\t", index_col=0)
expr = expr.T  # now rows = samples, cols = genes

# clinical file has a ton of columns, we just need the sample id + PAM50 call
# note: this file has no .gz in the name but might still be gzipped, so just try both
print("loading clinical data...")
try:
    clin = pd.read_csv(clin_path, sep="\t", index_col=0)
except UnicodeDecodeError:
    clin = pd.read_csv(clin_path, sep="\t", index_col=0, compression="gzip")
labels = clin["PAM50Call_RNAseq"]

# some samples wont have a label, just drop those
labels = labels.dropna()

# match samples that show up in both files
common_ids = expr.index.intersection(labels.index)
print("samples with both expression and label:", len(common_ids))

expr = expr.loc[common_ids]
labels = labels.loc[common_ids]

# stick the label on as the last column
df = expr.copy()
df["subtype"] = labels

# quick sanity check before saving
print(df["subtype"].value_counts())

df.to_csv(out_dir + "tcga_brca.csv")
print("saved to " + out_dir + "tcga_brca.csv")