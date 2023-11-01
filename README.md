# README #

This is the official Pytorch implementation of 
"[Operating critical machine learning models in resource constrained regimes](https://arxiv.org/abs/2302.06900)", Raghavendra Selvan et al. 2023

![results](utils/results.png)
### What is this repository for? ###

* Reproduce results on LIDC dataset reported in the paper
* v1.0

### How do I get set up? ###

* Basic Pytorch dependency
* Tested on Pytorch 1.7, Python 3.8
* The preprocessed LIDC data is provided as an archive [here](). Unzip and point that as the data_dir.
* Fine tune a pretrained Densenet model from with half precision using 8-bit optimizer: 

python train.py --batch_size 32 --timm --num_epochs 50 --lr 1e-5 --seed 1 --lidc --model_name densenet --bnb --half


### Usage guidelines ###

* Kindly cite our publication if you use any part of the code
```
@inproceedings{raghav2023Operating,
 	title={Operating critical machine learning models in resource constrained regimes},
	author={Raghavendra Selvan and Julian Schön and Erik B. Dam},
	booktitle={Workshop on Resource Aware Medical Imaging at MICCAI},
	month={July},
 	note={arXiv preprint arXiv:2302.06900},
	year={2023}}
```
### Who do I talk to? ###

* raghav@di.ku.dk

