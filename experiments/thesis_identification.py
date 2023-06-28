import os.path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.conversation import Conversation
from analysis.inference import query, prettify, cola


def random_sentence_from_conversation(conv, n_sentences=1):
    theses = []
    phrases = [expression.phrase for expression in conv.expressions]
    for _ in tqdm(range(n_sentences)):
        phrase = np.random.choice(phrases)
        phrase = prettify(phrase)
        sentences = phrase.split(".")
        sentence = np.random.choice(sentences)

        i = 0
        while len(sentence) < 5:
            sentences = sentences.remove(sentence)
            sentence = np.random.choice(sentences)

        theses.append(sentence)
        time.sleep(1)
    return theses


def manual_theses():
    return [
        "All lockdown measures taken were necessary and meaningfully contributed to the reduction of the spread of the virus.",
        "The rationale behind some lockdown measures was unclear and hard to understand.",
        "Regional differences between lockdown measures often led to people skirting restrictions.",
        "During the opening-up phase, many remaining restrictions seemed arbitrary and contradictory.",
        "Some technological measures introduced to fight the pandemic made life harder for elderly people and others who are not tech-savvy.",
        "Some hygiene measures introduced during the pandemic should become a new social norm.",
        "A complete lockdown should have been introduced earlier to fight the spread of the virus."
    ]


def random_sentence(n_sentences=20):
    theses = []
    for i in tqdm(range(n_sentences)):
        thesis = query("Generate a random sentence.")
        theses.append(thesis)
        # sleep one second
        time.sleep(1)
    return theses


def random_theses(n_sentences=20):
    theses = []
    for _ in tqdm(range(n_sentences)):
        thesis = query(
            "Give an argumentative thesis on a completely random subject. It should be a single sentence. Thesis:")
        theses.append(thesis)
        # sleep one second
        time.sleep(1)

    return theses


relvs = []
args = []

# for fname in tqdm(os.listdir("data/ami-corpus/vtt")):
path = "../data/ami-corpus/vtt/merged/ES2000.vtt"

model_name = "random-unrelated-v2"
# model_name = "dolly-12b"
n = 1
max_length = 2000
sleep = 1
folder_path = path.split(".")[0]
conversation = Conversation(path)
conversation.load(merge_consecutive=True)
# load csv with pd

# with open(os.path.join("../results/ami-corpus/llm", f"{model_name}-n={n}.csv")) as f:
#     df = pd.read_csv(f)
#     theses = df["thesis"].tolist()

# df = pd.read_csv(os.path.join("../results/ami-corpus/llm", f"{model_name}-n={n}.csv"), index=False)

theses = conversation.extract_theses_manually(n=n, n_tokens_per_chunk=1500)

# theses = conversation.extract_theses(model_name=model_name, n=n, n_tokens_per_chunk=1500, max_length=max_length,
#                                      truncate_thesis=False, sleep=sleep, save_prompts=True)
# theses = random_sentence_from_conversation(conversation, n_sentences=20)
# theses = random_sentence(n_sentences=20)
# theses = random_theses(n_sentences=20)
evaluations = [conversation.evaluate_relevance(thesis, k=10) for thesis in tqdm(theses)]
# argumentativeness = [get_argumentativeness(thesis) for thesis in tqdm(theses)]
# cola_judgement = [cola(thesis) for thesis in tqdm(theses)]
# cola_judgement = np.where(cola_judgement, True, False)

df = pd.DataFrame()
df["thesis"] = theses
df["evaluation"] = evaluations
# df["argumentativeness"] = argumentativeness
# df["cola"] = cola_judgement
df.to_csv(os.path.join("../results/ami-corpus/llm", f"{model_name}-n={n}.csv"), index=False)

print(np.mean(evaluations))
# print(np.mean(cola_judgement))
