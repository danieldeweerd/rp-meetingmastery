import numpy as np
from tqdm import tqdm

from theses import get_embeddings

# theses = get_theses()
# thesis_embeddings = get_minilm_embedding_batch(theses)
thesis_embeddings = get_embeddings()

sims = []

for i in tqdm(range(len(thesis_embeddings))):
    embedding = thesis_embeddings[i]
    crossval = np.concatenate([thesis_embeddings[:i],
                               thesis_embeddings[i + 1:]])
    sim = np.mean(np.matmul(crossval, embedding))
    sims.append(sim)

print(np.mean(sims))

# def get_argumentativeness(phrase):
# phrase_embedding = np.array(get_minilm_embedding(phrase))
# return np.mean(np.matmul(thesis_embeddings, phrase_embedding))
