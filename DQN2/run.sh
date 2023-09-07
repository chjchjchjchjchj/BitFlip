#!/bin/bash
for length in {4,8,16,32,64}; do
    python main.py length=$length exp_name="DQN in $length bits"
done