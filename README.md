# MDGCN
> Multi-Dependency Graph Convolutional Network with Cross-View Contrastive Learning for Drug Repositioning
- framework picture
- paper url

## Introduction

- abstract about model

## Environment
See our `environment.yml`

The codes of MDGCN are implemented and tested under the following development environment: **Python=3.7.16 | Torch=1.10.0+cu113 | Numpy=1.21.6 | Scipy=1.6.0**

## Run
Please clone or download this repository, then run `main.py`
The commands specify the hyperparameter settings that generate the reported results in the paper.

- Fdataset
```
python main.py --dataset Fdataset -- epochs 150 --batch 4096 --decay 0.97 --lr 0.067 --layers 5 -- rank 4 --topK 5 --ssl_reg_r 0.068 --ssl_reg_d 0.088 --wr1 1.0 --wr2 0.0 --wd1 1.0 --wd2 0.0
```
- Cdataset
```
python main.py --dataset Cdataset -- epochs 60 --batch 4096 --decay 0.99 --lr 0.055 --layers 8 -- rank 6 --topK 4 --ssl_reg_r 0.068 --ssl_reg_d 0.085 --wr1 0.7 --wr2 0.3 --wd1 0.7 --wd2 0.3
```
- LRSSL
```
python main.py --dataset lrssl -- epochs 70 --batch 8192 --decay 0.99 --lr 0.055 --layers 8 -- rank 6 --topK 6 --ssl_reg_r 0.08 --ssl_reg_d 0.09 --wr1 0.8 --wr2 0.2 --wd1 0.8 --wd2 0.2
```
- Ldataset
```
python main.py --dataset Ldataset -- epochs 60 --batch 4096 --decay 0.99 --lr 0.1 --layers 10 -- rank 4 --topK 5 --ssl_reg_r 0.064 --ssl_reg_d 0.085 --wr1 0.7 --wr2 0.3 --wd1 0.7 --wd2 0.3
```

## Datasets

| Dataset          | No. of Drugs | No. of Diseases | No. of Associations | Sparsity   |
|------------------| ------------ | --------------- |---------------------|------------|
| Fdataset/Gdataset | 593          | 313             | 1933                | 0.0104     |
| Cdataset         | 663          | 409             | 2532                | 0.0087     |
| LRSSL            | 763          | 681             | 3051                | 0.0059     |
| Ldataset(LAGCN)  | 269          | 598             | 18416               | 0.1145     |
> Data above from [AdaDR](https://github.com/xinliangSun/AdaDR/tree/main/AdaDR/raw_data/drug_data)

### Description

- [Fdataset](https://github.com/BioinformaticsCSU/BNNR)/Gdataset (Gottlied et.)
  - Gottlieb A, Stein GY, Ruppin E, et al. PREDICT: a method for inferring novel drug indications with application to personalized medicine. Mol Syst Biol 2011;7:496.
- [Cdataset]((https://github.com/BioinformaticsCSU/BNNR))
  - Luo H, Wang J, Li M, et al. Drug repositioning based on comprehensive similarity measures and bi-random walk algorithm. Bioinformatics 2016;32:2664–71.
- [LRSSL](https://github.com/linwang1982/DRIMC)
  - Liang X, Zhang P, Yan L, et al. LRSSL: predict and interpret drug–disease associations based on data integration using sparse subspace learning. Bioinformatics 2017;33:1187–96.
- [Ldataset](https://github.com/storyandwine/LAGCN)/LAGCN
  - Yu Z, Huang F, Zhao X, et al. Predicting drug–disease associations through layer attention graph convolutional network. Brief Bioinform 2020;22:1–11.

## Result

## Citation

If you find this work helpful to your research, please kindly consider citing our paper.
```
@inproceedings{xx,
  title={xx},
  author={xx},
  booktitle={xx},
  pages={xxx--xx},
  year={2025}
}
```
---
## Experiment

### Performance of model in cross-validation

We execute 10-fold cross-validation to evaluate the performance of model.
During the 10-fold cross-validation, all known and unknown drug–disease associations
are randomly divided into 10 exclusive subsets of approximately equal size, respectively.
Each subset is treated as the testing set in turn,
while the remaining nine subsets are used as the training set.
Then, the area under the receiver operating characteristic curve (AU-ROC)
and the area under the precision-recall curve (AU-PRC) are adopted to measure the overall performance of model.

### Predicting indications for new drugs

For each drug $r_i$,we delete all known drug–disease associations about drug $r_i$ as the testing set and use all the remaining associations as the training samples.

### Parameter analysis

### Ablation study

### Case studies

we apply model to predict candidate drugs for two diseases including **Alzheimer’s disease (AD)** and **Breast carcinoma (BRCA)**.

- AD is a progressive neurological degenerative disease that has no efficacious medications available yet.

- BRCA is a phenomenon in which breast epithelial cells proliferate out of control under the action of a variety of oncogenic factors. *Although there are many drugs for breast cancer, such as Paclitaxel, Carboplatin and so on, a wider choice of drugs may provide better treatment options.*

During the process, **all the known** drug–disease associations in the Fdataset are treated **as the training set** and **the missing** drug–disease associations regarded **as the candidate set**. After all the missing drug–disease associations are predicted, we subsequently rank the candidate drugs by the predicted probabilities for each drug. We focus on the top five potential drugs for breast carcinoma and AD and adopt highly reliable sources (i.e. CTD and PubMed) to check the predicted drug–disease associations.
