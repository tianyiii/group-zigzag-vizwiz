import numpy as np
import matplotlib as plt
import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

data_path = "qual_ann/train.json"
with open(data_path) as f:
    data = json.load(f)

flaws = defaultdict(int)
valid_count = 0
print(len(data))
for ele in data:
    # only check recognizable images
    if ele["unrecognizable"] < 2:
        valid_count += 1
        for k, v in ele["flaws"].items():
            if v > 1:
                flaws[k] += 1

flaw = {}
for k, v in flaws.items():
    flaw[k] = float(v) / valid_count
print(valid_count)
print(flaw)
print(flaws)

plt.pie([float(v) for v in flaws.values()], labels=[_ for _ in flaws.keys()],
           autopct=None)

plt.draw()
plt.savefig("flaws.png")
plt.close()
