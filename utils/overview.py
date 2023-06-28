import json
import os

path = "data/ami-corpus/extractive/"
fnames = os.listdir(path)

for fname in fnames:
    if not fname == "ES2002a.json":
        # print(fname)
        continue

    fpath = os.path.join(path, fname)
    with open(fpath, 'r') as f:
        data = json.load(f)

    for d in data:
        print(d["text"])
