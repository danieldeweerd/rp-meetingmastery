
# Load "data/raw/group5_session2_video.vtt" as a text file

from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from ai_utils import get_minilm_embedding_batch, get_ada_embedding_batch


transcript_raw = open("data/raw/group5_session2_video.vtt", "r")
transcript_processed = open("data/processed/group5_session2_video.txt", "w")

lines = transcript_raw.readlines()
lines_processed = []
embeddings = []

i = 1
for line in lines:
    if line.startswith("["):
        line = line.strip()
        name = line[0:line.index("]") + 1]
        name_index = int(name[-3])
        relevant = line[line.index("]") + 1:]

        speaker_tag = "SPEAKER {}: ".format(name_index)

        transcript_processed.write(str(i) + " " + speaker_tag + relevant + "\n")
        lines_processed.append(relevant)
        i += 1

transcript_raw.close()
transcript_processed.close()

embeddings = get_minilm_embedding_batch(lines_processed)
embeddings = np.array(embeddings)
print(embeddings.shape)

# tsne = TSNE()
# X_new = tsne.fit_transform(embeddings)
# print(X_new.shape)
# plt.scatter(X_new[:, 0], X_new[:, 1])
# plt.show()

similarity_matrix = np.inner(embeddings, embeddings)

indx = 10
sentence = lines_processed[indx]
sims = []
for i in range(100):
    mindx = np.argmax(similarity_matrix[indx])
    sims.append(lines_processed[mindx])
    similarity_matrix[indx][mindx] = 0

print(sentence)
for sim in sims:
    print(sim)
    print("\n")


# compute length of all embeddings
# embedding_lengths = np.linalg.norm(embeddings, axis=1)
# print(embedding_lengths)
# plt.hist(embedding_lengths, bins=100)
# plt.show()


print("Mean sentence length in characters:", np.mean([len(line) for line in lines_processed]))