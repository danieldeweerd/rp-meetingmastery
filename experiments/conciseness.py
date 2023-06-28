import os

import pandas as pd

path = "../results/ami-corpus/llm"

for file in os.listdir(path):
    n_words = []
    n_chars = []
    n_sentences = []

    if file.endswith(".csv"):
        with open(os.path.join(path, file), "r") as f:
            df = pd.read_csv(f)
        theses = df["thesis"]
        for thesis in theses:
            thesis = str(thesis)
            n_words.append(len(thesis.split(" ")))
            n_chars.append(len(thesis))
            splits = thesis.split(".")

            if len(splits) == 1:
                n_sentences.append(1)
                continue

            if len(splits[1]) > 1:
                n_sentences.append(len(thesis.split(".")))
            else:
                n_sentences.append(1)

        # print stats
        # print(r"LAM & {} & {} & {} & {}\\".format(file, sum(n_words)/len(n_words), sum(n_chars)/len(n_chars), sum(n_sentences)/len(n_sentences)))

        # print stats rounded to 1 decimal
        print(r"LAM & {} & {} & {} & {}\\".format(file.split("-n=")[0], int(round(sum(n_chars) / len(n_chars), 0)),
                                                  int(round(sum(n_words) / len(n_words), 0)),
                                                  round(sum(n_sentences) / len(n_sentences), 1)))

        # print("File: {} | n_words: {} | n_chars: {} | n_sentences: {}"
