#!/bin/bash

# 循环从2到50
model=DQN
cd /root/autodl-tmp/ling/DQN
# for length in {2..10}; do
#     # 执行命令，将当前的length值传递给DQN.py
#     python main.py length=$length exp_name="$length bits"
# done

# for length in {10..20}; do
#     # 执行命令，将当前的length值传递给DQN.py
#     python main.py length=$length exp_name="$length bits"
# done

# for length in {20..30}; do
#     # 执行命令，将当前的length值传递给DQN.py
#     python main.py length=$length exp_name="$length bits"
# done

for length in {30..40}; do
    # 执行命令，将当前的length值传递给DQN.py
    python main.py length=$length exp_name="$length bits"
done