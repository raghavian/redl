#!/bin/bash
set -x 
epochs=50
for s in {1,10,100}
do
for m in {'densenet121','swinv2_base_window8_256','vit_base_patch32_plus_256'}
    do 
    python train.py --batch_size 32 --timm --num_epochs $epochs --lr 1e-5 --seed $s --lidc --model_name $m
    python train.py --batch_size 32 --timm --num_epochs $epochs --lr 1e-5 --seed $s --lidc --model_name $m --bnb
    python train.py --batch_size 32 --timm --num_epochs $epochs --lr 1e-5 --seed $s --lidc --model_name $m --bnb --half
    python train.py --batch_size 32 --timm --num_epochs $epochs --lr 1e-5 --seed $s --lidc --model_name $m --amp
    python train.py --batch_size 32 --timm --num_epochs $epochs --lr 1e-5 --seed $s --lidc --model_name $m --amp --bnb
    done
done
