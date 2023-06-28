import numpy as np

from ai_utils import get_embedding
from conversation import Conversation
from theses import get_argumentativeness


def semantic_similarity(string_1, string_2, use_ada=False):
    embedding_1 = get_embedding(string_1, use_ada=use_ada)
    embedding_2 = get_embedding(string_2, use_ada=use_ada)

    return np.dot(embedding_1, embedding_2)


conv = Conversation("data/memo-corpus/vtt/group5_session2_video.vtt")
conv.load()

while True:
    thesis = input("Enter thesis: ")

    top_k_sim = conv.evaluate_relevance(thesis, ignore_top=True)
    arg_sim = get_argumentativeness(thesis)

    print("Top k similarity: {}".format(top_k_sim))
    print("Argumentativeness similarity: {}".format(arg_sim))


