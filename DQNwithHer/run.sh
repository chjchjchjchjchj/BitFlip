#!/bin/bash

# 循环从2到50
model=DQNwithHER
# for length in {4..10}; do
#     python main.py minimal_size=128 length=$length use_wandb=false
# done

# for length in {10..20}; do
#     python main.py minimal_size=128 length=$length use_wandb=false
# done

# for length in {20..30}; do
#     python main.py minimal_size=128 length=$length use_wandb=false
# done

# for length in {30..40}; do
#     python main.py minimal_size=128 length=$length use_wandb=false
# done


python main.py minimal_size=128 length=6 use_wandb=false 
python main.py minimal_size=128 length=7 use_wandb=false
python main.py minimal_size=128 length=8 use_wandb=false