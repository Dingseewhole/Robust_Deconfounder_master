# Robust Deconfounder
This is a pytorch implementation of Robust Deconfounder (RD), a robust debiasing method for recommendation system. Robust Deconfounder is proposed in the paper:

[Addressing Unmeasured Confounder for Recommendation with Sensitivity Analysis]

by  Sihao Ding, Peng Wu, Fuli Feng, Xiangnan He, Yitong Wang, Yong Liao and Yongdong Zhang

Published at SIGKDD 2022. If you use this code please cite our paper.

## Introduction

RD is a method that combats the unmeasured confounder based on propensity-based recommender models. We use sensitivity analysis to estimate the effect of unmeasured confounders, and employ the adversarial learning mechanism to train a model that more robusts to unmeasured confounders.

## Environment Requirement

The code runs well under python 3.8.10. The required packages are as follows:

- pytorch == 1.7.1
- numpy == 1.19.1
- scipy == 1.5.2
- pandas == 1.1.3
- cppimport == 20.8.4.2
- tqdm == 4.62.3 

## Datasets
We use public/private dataset. 

- user.txt: biased data collected by normal policy of recommendation platform. For Yahoo!R3 and Coat, each line is user ID, item ID, rating of the user to the item. For Simulation, each line is user ID, item ID, position of the item, binary rating of the user to the item. 
- random.txt: unbiased data collected by stochastic policy where items are assigned to users randomly. Each line in the file is user ID, item ID, rating of the user to the item. 

## Run the Code
You can quickly reproduced the results of IPS on Yahoo!R3 or Coat with the command:
```shell
python IPS.py --dataset yahooR3 or coat
```
You can quickly reproduced the results of Robust Deconfounder IPS (RD-IPS) on Yahoo!R3 or Coat with the command:
```shell
python IPS_RD.py --dataset yahooR3 or coat
```
You can quickly reproduced the results of Benchmarked Robust Deconfounder IPS (BRD-IPS) on Yahoo!R3 or Coat with the command:
```shell
python IPS_BRD.py --dataset yahooR3 or coat
```

You can quickly reproduced the results of Doubly Robust (DR) on Yahoo!R3 or Coat with the command:
```shell
python DR.py --dataset yahooR3 or coat
```
You can quickly reproduced the results of Robust Deconfounder Doubly Robust (RD-DR) on Yahoo!R3 or Coat with the command:
```shell
python DR_RD.py --dataset yahooR3 or coat
```
You can quickly reproduced the results of Benchmarked Robust Deconfounder DR (BRD-DR) on Yahoo!R3 or Coat with the command:
```shell
python DR_BRD.py --dataset yahooR3 or coat
```

You can quickly reproduced the results of AutoDebias on Yahoo!R3 or Coat with the command:
```shell
python Autodebias.py --dataset yahooR3 or coat
```
You can quickly reproduced the results of Robust Deconfounder AutoDebias (RD-AutoDebias) on Yahoo!R3 or Coat with the command:
```shell
python Autodebias_RD.py --dataset yahooR3 or coat
```
You can quickly reproduced the results of Benchmarked Robust Deconfounder AutoDebias (BRD-AutoDebias) on Yahoo!R3 or Coat with the command:
```shell
python Autosebias_BRD.py --dataset yahooR3 or coat
```

To implement this code with other teacher models or with on other datasets you may need to fine-tune the hypermenters, and you can find out all hypermenters you need in _arguments.py_.

## Acknowledgment
Some parts of this repository are adopted from AutoDebias and Meta-learning, you can find more information in https://github.com/DongHande/AutoDebias and https://github.com/AdrienLE. Thanks for the contributions!
