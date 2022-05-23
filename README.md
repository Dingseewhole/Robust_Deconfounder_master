# InterD
This is the official pytorch implementation of InterD, a debiasing method for recommendation system. InterD is proposed in the paper:

[Interpolative Distillation for Unifying Biased and Debiased Recommendation]

by  Sihao Ding, Fuli Feng, Xiangnan He, Jinqiu Jin, Wenjie Wang, Yong Liao and Yongdong Zhang

Published at SIGIR 2022. If you use this code please cite our paper.

## Introduction

InterD is a method that unifies biased and debiased methods as teachers to ahcieve strong performance on both normal biased test and debiased test with alleviating over-debiased issue and bias amplification issue in recommendation.

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
You can quickly reproduced the results on Yahoo!R3 with the default parameters by the command:
```shell
python train_explicit.py --dataset yahooR3
```
To implement this code with other teacher models or with on other datasets you may need to fine-tune the hypermenters, and you can find out all hypermenters you need in _arguments.py_.

## Acknowledgment
Some parts of this repository are adopted from AutoDebias and Meta-learning, you can find more information in https://github.com/DongHande/AutoDebias and https://github.com/AdrienLE.
