import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from analysis.ai_utils import get_minilm_embedding
from analysis.conversation import Conversation

theses_path = "../data/ami-corpus/vtt/merged/theses.csv"
conversation_path = "../data/ami-corpus/vtt/merged/ES2000.vtt"
output_root_path = "../data/ami-corpus/vtt/merged/disagreements/"
theses = pd.read_csv(theses_path)["thesis"].tolist()
conversation = Conversation(conversation_path)
conversation.load(merge_consecutive=True)

for thesis in theses:
    if os.path.exists(output_root_path + thesis.replace(" ", "_") + ".txt"):
        continue
    thesis_embedding = get_minilm_embedding(thesis)
    sim_values = np.dot(conversation.embeddings, thesis_embedding)
    window_size = 100
    sim_values = np.convolve(sim_values, np.ones(100) / 100, mode='same')

    # find top-2 peaks in sim_values signal
    peaks, _ = find_peaks(sim_values)
    peaks = peaks[np.argsort(sim_values[peaks])][::-1][:2]
    peaks = np.sort(peaks)
    plt.axvline(peaks[0], color="red")
    index_0 = peaks[0]
    relevant_expressions = conversation.expressions[index_0 - 20:index_0 + 20]
    # relevant_expressions = relevant_expressions[len(relevant_expressions)]

    plt.plot(sim_values)
    plt.title(thesis)
    plt.show()
    # sims = conversation.get_most_similar(thesis, n=20)
    template = "Task: for the given argumentative thesis and dialogue, indicate for every sentence whether it agrees with, " \
               "disagrees with, or is unrelated to the thesis. Give your response as a Python list per sentences, " \
               "where 0 means unrelated, 1 means agree, and -1 means disagree.\n " \
               "Thesis: {}\n" \
               "Dialogue:\n"

    print("THESIS:", thesis)
    print("DIALOGUE:")
    classifications = []
    for i, s in enumerate(relevant_expressions):
        print("\t", "{}. [Speaker {}]: {}".format(i + 1, s.participant_id, s.phrase))
        c = input("Agree (a), disagree (d), or unrelated ([blank])?")
        if c == "a":
            classifications.append(1)
        elif c == "d":
            classifications.append(-1)
        else:
            classifications.append(0)
    # save list
    with open(output_root_path + thesis.replace(" ", "_") + ".txt", "w") as f:
        f.write(str(classifications))
