import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path = "../results/ami-corpus/llm"

name = []
relevance = []
argumentative = []
colas = []

files = os.listdir(path)
for file in files:
    # if "lam-" not in file:
    #     continue
    frame = pd.read_csv(path + "/" + file)
    name.append(file.split("-theses")[0].split("-n")[0])
    relevance.append(frame["evaluation"].mean())
    argumentative.append(frame["argumentativeness"].mean())
    if "cola" in frame:
        colas.append(frame["cola"].mean())
    else:
        colas.append(0)

# argumentative = [x 64 for x in argumentative]

plt.bar(name, relevance)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.title("Relevance")
plt.ylim(0.3, 0.6)
plt.show()

plt.bar(name, argumentative)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.title("Argumentativeness")
plt.show()

plt.bar(name, colas)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.title("CoLA")
plt.show()

print(colas)

for i, nm in enumerate(name):
    print(nm, colas[i])
    # fmt = r"LAM & {} & {} & {} \\".format(nm, np.round(relevance[i], 3), np.round(argumentative[i], 3))
    # fmt = fmt.replace("!", "{")
    # fmt = fmt.replace("$", "}")
    # print(fmt)
    # print(nm, relevance[i], argumentative[i])
