import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import fkassim.FastKassim as fkassim

FastKassim = fkassim.FastKassim(fkassim.FastKassim.LTK)
frame = pd.read_csv("baseline_theses.csv")
ref_theses = frame["theses"].tolist()[:50]
ref_theses_trees = [FastKassim.parse_document(ref_thesis) for ref_thesis in tqdm(ref_theses, desc="Parsing trees")]


def get_sas(thesis):
    thesis_parsetree = FastKassim.parse_document(thesis)
    similarities = []
    for ref_parsetree in ref_theses_trees:
        similarity = FastKassim.compute_similarity_preparsed(ref_parsetree, thesis_parsetree)
        similarities.append(similarity)
    return np.mean(similarities)


csvs = os.listdir("stuff")
for csv in csvs:
    with open(os.path.join("stuff", csv)) as f:
        df = pd.read_csv(f)
        theses = df["thesis"].tolist()
        sas = [get_sas(thesis) for thesis in tqdm(theses, desc=csv)]
        print(csv, np.mean(sas))
        df["sas"] = sas
        df.to_csv(os.path.join("stuff", csv), index=False)