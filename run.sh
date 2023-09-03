#!/bin/bash

# 循环从2到50
for length in {2..50}; do
    # 执行命令，将当前的length值传递给DQN.py
    python DQN.py length=$length
done
