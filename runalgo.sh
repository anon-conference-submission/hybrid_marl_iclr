#!/bin/bash

for i in {0..4}
do
   python3 src/main.py --config=iql_ns --env-config=gymma with env_args.key="SimpleSpread-v0" seed=$i &
   echo "Running with seed=$i"
   sleep 2s
done
