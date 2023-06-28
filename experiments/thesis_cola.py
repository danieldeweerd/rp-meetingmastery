import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.inference import continuous_cola

path = "../results/ami-corpus/llm"
# args_path = "../results/ami-corpus/llm/random-argumentative-n=1.csv"

# for th in tqdm(pd.read_csv(args_path)["thesis"]):
#     print(th, cola(th))
#

for file in os.listdir(path):
    # if "lam-" in file or "random-" in file:
    #     continue

    # if "stablelm-" not in file:
    #     continue

    frame = pd.read_csv(os.path.join(path, file))
    colas = []
    for th in tqdm(frame["thesis"]):
        colas.append(continuous_cola(th))

    print(file, "|", np.mean(colas))
    # frame["cola"] = colas
    # frame.to_csv(os.path.join(path, file), index=False)
