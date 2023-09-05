#!/bin/bash
cd /root/BitFlip/DQN
# python main.py length=4 use_wandb=false
python main.py length=4 epsilon=0.5


cd /root/BitFlip/DQNwithHer
python main.py length=4 epsilon=0.9 delta_epsilon=1e-5

cd /root/BitFlip/DQNwithHer
python main.py length=4 epsilon=0.5 delta_epsilon=1e-5 target_update=10
python main.py length=4 epsilon=0.5 delta_epsilon=1e-5 target_update=50 # 50效果最好

python main.py length=4 epsilon=0.3 delta_epsilon=1e-5 target_update=10



python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50

python main.py length=4 epsilon=0.8 delta_epsilon=1e-5 target_update=50 
python main.py length=4 epsilon=0.7 delta_epsilon=1e-5 target_update=50 
python main.py length=4 epsilon=0.6 delta_epsilon=1e-5 target_update=50 
python main.py length=4 epsilon=0.5 delta_epsilon=1e-5 target_update=50 
python main.py length=4 epsilon=0.4 delta_epsilon=1e-5 target_update=50 
python main.py length=4 epsilon=0.3 delta_epsilon=1e-5 target_update=50 
python main.py length=4 epsilon=0.2 delta_epsilon=1e-5 target_update=50 
python main.py length=4 epsilon=0.1 delta_epsilon=1e-5 target_update=50 


python main.py length=4 epsilon=0.1 delta_epsilon=1e-5 target_update=50 
python main.py length=10 epsilon=0.1 delta_epsilon=1e-5 target_update=50 
python main.py length=15 epsilon=0.1 delta_epsilon=1e-5 target_update=50 
python main.py length=20 epsilon=0.1 delta_epsilon=1e-5 target_update=50 
python main.py length=25 epsilon=0.1 delta_epsilon=1e-5 target_update=50 
python main.py length=30 epsilon=0.1 delta_epsilon=1e-5 target_update=50 
python main.py length=35 epsilon=0.1 delta_epsilon=1e-5 target_update=50 
python main.py length=40 epsilon=0.1 delta_epsilon=1e-5 target_update=50 
python main.py length=45 epsilon=0.1 delta_epsilon=1e-5 target_update=50 
python main.py length=50 epsilon=0.1 delta_epsilon=1e-5 target_update=50 


python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 

