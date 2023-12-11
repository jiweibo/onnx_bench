import numpy as np
import argparse
import os


data1 = "data_fp16_local3"
data2 = "data_fp16_local4"

item1 = os.listdir(data1)
item2 = set(os.listdir(data2))

common_item = []
for n in item1:
  if n in item2:
    common_item.append(n)

base_cmd = "python3 ../../check.py"

for n in common_item:
  print("Processing ", n)
  os.system(base_cmd + " " + os.path.joinpath(data1, n) + " " + os.path.joinpath(data2, n))