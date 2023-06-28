import os

import numpy as np
import pandas as pd

from analysis.ai_utils import get_minilm_embedding_batch

path = "../results/ami-corpus/llm"

for file in os.listdir(path):
    n_words = []
    n_chars = []
    n_sentences = []

    if file.endswith(".csv"):
        with open(os.path.join(path, file), "r") as f:
            df = pd.read_csv(f)
        theses = df["thesis"].tolist()
        theses = [str(thesis) for thesis in theses]
        embeddings = get_minilm_embedding_batch(theses)
        embeddings = np.array(embeddings)
        print(file, np.var(embeddings))
        for thesis in theses:
            pass
