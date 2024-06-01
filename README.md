# MDGCN
> > Multi-Dependency Graph Convolutional Network with Cross-View Contrastive Learning for Drug Repositioning

## Golden Standard Datasets

- Fdataset/Gdataset (Gottlied et.)
  - Gottlieb A, Stein GY, Ruppin E, et al. PREDICT: a method for inferring novel drug indications with application to personalized medicine. Mol Syst Biol 2011;7:496.
- Cdataset
  - Luo H, Wang J, Li M, et al. Drug repositioning based on comprehensive similarity measures and bi-random walk algorithm. Bioinformatics 2016;32:2664–71.
- LRSSL
  - Liang X, Zhang P, Yan L, et al. LRSSL: predict and interpret drug–disease associations based on data integration using sparse subspace learning. Bioinformatics 2017;33:1187–96.
- Ldataset/LAGCN
  - Yu Z, Huang F, Zhao X, et al. Predicting drug–disease associations through layer attention graph convolutional network. Brief Bioinform 2020;22:1–11.

### Description of DRHGCN

* Fdataset and Cdataset https://github.com/BioinformaticsCSU/BNNR
* LRSSL https://github.com/linwang1982/DRIMC
* Ldataset https://github.com/storyandwine/LAGCN
* HDVD https://github.com/luckymengmeng/SCPMF

### Details of dataset
> Data below can be downloaded from [AdaDR](https://github.com/xinliangSun/AdaDR/tree/main/AdaDR/raw_data/drug_data)

| Dataset          | No. of Drugs | No. of Diseases | No. of Associations | Sparsity   |
|------------------| ------------ | --------------- |---------------------|------------|
| Fdataset/Gdataset | 593          | 313             | 1933                | 0.0104     |
| Cdataset         | 663          | 409             | 2532                | 0.0087     |
| LRSSL            | 763          | 681             | 3051                | 0.0059     |
| Ldataset(LAGCN)  | 269          | 598             | 18416               | 0.1145     |

# Experiment

## Performance of model in cross-validation

We execute 10-fold cross-validation to evaluate the performance of model.
During the 10-fold cross-validation, all known and unknown drug–disease associations
are randomly divided into 10 exclusive subsets of approximately equal size, respectively.
Each subset is treated as the testing set in turn,
while the remaining nine subsets are used as the training set.
Then, the area under the receiver operating characteristic curve (AUROC)
and the area under the precision-recall curve (AUPRC) are adopted to measure the overall performance of model.

## Predicting indications for new drugs

For each drug $r_i$,we delete all known drug–disease associations about drug $r_i$ as the testing set and use all the remaining associations as the training samples.

## Parameter analysis

## Ablation study

## Case studies

we apply model to predict candidate drugs for two diseases including **Alzheimer’s disease (AD)** and **Breast carcinoma (BRCA)**.

- AD is a progressive neurological degenerative disease that has no efficacious medications available yet.

- BRCA is a phenomenon in which breast epithelial cells proliferate out of control under the action of a variety of oncogenic factors. *Although there are many drugs for breast cancer, such as Paclitaxel, Carboplatin and so on, a wider choice of drugs may provide better treatment options.*

During the process, **all the known** drug–disease associations in the Gdataset are treated **as the training set** and **the missing** drug–disease associations regarded **as the candidate set**. After all the missing drug–disease associations are predicted, we subsequently rank the candidate drugs by the predicted probabilities for each drug. We focus on the top five potential drugs for breast carcinoma and AD and adopt highly reliable sources (i.e. CTD and PubMed) to check the predicted drug–disease associations.
