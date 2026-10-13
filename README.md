# Does a patient-similarity graph help predict breast cancer subtypes?

A small study on TCGA-BRCA gene expression, using graph neural networks (GCN and GraphSAGE) and comparing them against models that don't use a graph.

I started this project expecting the graph to help. It mostly didn't. I kept the negative result in, because working out *why* turned out to be the most interesting part.

## About this project

This is an independent project that I built and ran on my own laptop (Windows, CPU only, no GPU), as a way to learn graph neural networks on real biomedical data. I used an AI assistant (Claude) for coding help and to talk through results. I ran every experiment myself, the numbers below are copied from my own terminal outputs, and I can explain what each script does.

## The question

PAM50 is a way of splitting breast tumors into 5 intrinsic subtypes (LumA, LumB, Her2, Basal, Normal-like) from the expression of 50 genes. I wanted to know:

1. If I connect patients with similar gene expression into a graph, does a GNN predict the subtype better than a plain model?
2. If yes, when? (for example, when only few patients have labels)
3. Which genes separate the subtypes, and can I trust that list?

## Data

- TCGA-BRCA from UCSC Xena: `HiSeqV2.gz` (RNA-seq expression, 20,530 genes) and the clinical matrix (column `PAM50Call_RNAseq`).
- 956 patients have both expression and a subtype label: LumA 434, LumB 194, Basal 142, Normal 119, Her2 67.
- The data is not in this repo (too big, and better downloaded from the source). Put the two raw files in the project folder and run `scripts/0_prepare_data.py`.

## How it works

- Every patient is a node. Node features are the expression values of the selected genes.
- Edges: each patient is connected to its 10 nearest neighbours (k-NN on standardized expression), made undirected. About 82% of the edges connect two patients with the same subtype.
- The graph is built on all patients together (transductive setting), but the labels of the test patients are never used for training.
- Gene filtering (variance filter, then top 1000 genes by ANOVA F-test) and standardization are fitted on the training part only, inside each cross-validation fold. I added this carefully because doing the F-test on all data would leak label information into the test set.
- Models: 2-layer GCN, 2-layer GraphSAGE, and two baselines without any graph (logistic regression, MLP). Hidden size 64, dropout 0.5, Adam with lr 0.01. I did not tune these.

## Results

### Main comparison (5-fold cross-validation, `9_cross_validation.py`)

| model | accuracy | macro F1 | Her2 F1 |
|---|---|---|---|
| Logistic regression | 0.903 ± 0.020 | 0.893 ± 0.024 | 0.818 ± 0.058 |
| GCN | 0.896 ± 0.019 | 0.888 ± 0.021 | 0.811 ± 0.049 |
| GraphSAGE | 0.907 ± 0.009 | 0.893 ± 0.012 | 0.789 ± 0.054 |

The three models are basically tied. The per-fold differences between GCN and logistic regression go both ways.

### Is it the graph? (`13_graph_controls.py`)

Same pipeline, same folds, macro F1:

| model | macro F1 |
|---|---|
| Logistic regression | 0.893 ± 0.024 |
| MLP (no edges) | 0.890 ± 0.024 |
| GCN, real k-NN edges | 0.888 ± 0.021 |
| GCN, random edges | 0.424 ± 0.034 |

- The GCN with the real graph is not better than an MLP without edges (mean difference -0.002).
- With random edges (same graph, but patient ids shuffled) performance collapses. So the network really does use its neighbours, and the k-NN graph connects the right patients. It just gives no information that the gene values don't already have.
- My explanation, which I did not prove: PAM50 labels are computed from the expression of 50 genes, so the labels are almost a function of the input features, and neighbours have little left to add.

### What if only few patients have labels? (`10_when_does_graph_help.py`)

Macro F1 on the unlabeled patients, 3 random splits per setting:

| labeled fraction | 5% | 10% | 20% | 50% | 80% |
|---|---|---|---|---|---|
| Logistic regression | 0.777 | 0.843 | 0.858 | 0.887 | 0.894 |
| GCN | 0.809 | 0.849 | 0.852 | 0.867 | 0.870 |
| GraphSAGE | 0.807 | 0.851 | 0.854 | 0.881 | 0.880 |

With 5% labels the graph models are about 3 points better, with more labels they are equal or slightly worse. The standard deviation at 5% is 0.03 to 0.05 and I only did 3 repeats, so I do not count this as a confirmed effect.

Two more checks from the same script:
- When I remove the 50 PAM50 genes from the input, the results barely move (logistic regression 0.836 at 10% labels and 0.893 at 80%). The subtype signal is spread over many other genes.
- Changing k from 3 to 40 changes macro F1 by less than about 0.02.

### Which genes matter? (`12_gene_ranking.py`)

My first idea was to read the gene importance from the models (logistic regression coefficients and GCN gradients, `11_stable_genes.py`). It did not work well: the top-20 lists changed a lot between folds (Jaccard similarity 0.25 and 0.11) and PAM50 genes were only slightly above chance. Many genes are strongly correlated, so a model can spread the weight over them in different ways each time.

So for the final list I used something much simpler: a Welch t-test, one subtype against all the others, on the training part of each fold.

- PAM50 genes in the top 20 of a subtype: 2.04 on average, versus 0.09 expected by chance.
- Overlap between folds (Jaccard): 0.59.
- LumA: ESR1, MAPT, NAT1, SCUBE2, AGR3 and ABAT go up, which fits with LumA being estrogen-receptor positive. The Basal list is mostly the opposite picture, with luminal genes such as TFF1, XBP1, AGR3 and NAT1 being low.
- LumB: SFRP1, MIA and FOXC1 (basal-like genes) are low, ESPL1 (cell cycle) is high.
- The Normal-like list is full of stroma-related genes (CLDN11, MAMDC2, ITM2A). My guess is that these samples contain a lot of normal or stromal tissue. I did not test this.
- Full table: `results/stable_genes_ttest.csv`.

## What went wrong along the way

I want to be honest about this, because most of what I learned came from things that failed.

- **Environment.** Google Colab is not reachable for me, Drive mounting failed, and Kaggle did not give me a GPU, so I ended up running everything locally on CPU. `pip` kept timing out, so I installed through a mirror and installed PyTorch from a wheel I downloaded in the browser.
- **Junk genes in the first interpretation.** My first model (all 11,364 genes after the variance filter) gave a top list of obscure genes like FAM157A and SPDYE1 and no PAM50 genes at all. I checked the raw data and found that many of those genes are almost never expressed. A plain ANOVA test found ESR1, FOXC1 and MLPH right away, so the data was fine and the problem was how I read importance from the model. That is why the pipeline now selects 1000 genes by F-test (on training data only) before training.
- **A wrong conclusion from one split.** On a single train/test split, logistic regression looked clearly better than the GCN, and Her2 looked like a big weakness of the GNN. With 5-fold cross-validation both claims disappeared. Since then I only report cross-validation numbers.
- **Slow GraphSAGE.** With 1000 features, GraphSAGE took about 90 seconds per run on my CPU, while the GCN took about 10. I precomputed the neighbour mean once (it does not change during training) and wrote the first layer by hand. I checked that it gives exactly the same output as the standard `SAGEConv` layer (maximum difference 0.0). This faster version is used in script 10.
- **A mistake in my PAM50 gene list.** I first typed the 50 genes from memory and MYC was missing, while two old gene names (KNTC2, CDCA1) were in there twice under other names. I fixed it using the published list and re-ran the gene ranking. The other long runs (scripts 10 and 11) were not repeated; they only use the list for the PAM50-removal check and the overlap counts, so these numbers may differ by a tiny amount.
- **Unstable gene importance** (see above). The honest result there is that model-based importance on correlated gene data is not something I would trust.

## Limitations

- The labels are computed from the same genes that I use as input. High accuracy is therefore not surprising, and this is probably the main reason why the graph adds nothing.
- One dataset, no external validation cohort.
- Her2 has only 67 patients, so its F1 is noisy (std about 0.05 across folds).
- The experiment with few labels has only 3 repeats per setting. It also uses a fixed number of epochs without early stopping, unlike scripts 9 and 13.
- Hyperparameters were not tuned. Different settings might change the picture a bit.
- The gene lists show statistical differences between groups. I did not do any pathway analysis and I make no claim about causes.
- For Basal only a few genes stay in the top 20 across folds, because many genes separate it equally well and their order changes.

## What I would try next

The control experiments say that the graph does not help much for PAM50, since the labels come from the same genes. A question I find more interesting is the boundary between subtypes: are LumA and LumB really two separate groups, or one continuum? The patient graph could be used to find patients that sit between subtypes, and then I could check if they differ in survival or in proliferation genes from the core patients of each subtype. **I have not done this yet.**

## How to run

```
pip install -r requirements.txt
python scripts/0_prepare_data.py           # needs HiSeqV2.gz and the clinical matrix in the project folder
python scripts/1_preprocess.py
python scripts/2_build_graph.py
python scripts/3_train_model.py
python scripts/4_evaluate.py
python scripts/5_interpret.py
python scripts/9_cross_validation.py       # about 10 min on my CPU
python scripts/10_when_does_graph_help.py  # about 8 min
python scripts/12_gene_ranking.py          # seconds
python scripts/13_graph_controls.py        # about 2.5 min
```

Run everything from the project root folder, the paths are relative. Scripts 6, 7, 8 and 11 are side experiments (F-test baseline, model comparison on one split, GCN variants, model-based gene importance). I kept them because they show the path that led to the final analysis.

## Folder

- `scripts/` numbered scripts, each one runs on its own
- `results/` final tables and plots (cross-validation, controls, label-fraction curve, stable genes)
- `results/stable_genes_ttest.csv` stable marker genes per subtype
- `results_all_genes/` outputs of my first run, before gene selection: the model used all 11,364 genes that passed the variance filter (test accuracy 0.844). I kept them to compare with the final pipeline, and because their top genes were the junk genes I describe above
- `requirements.txt`
  
## Tools

Python, pandas, numpy, scikit-learn, matplotlib, PyTorch, PyTorch Geometric.
