import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.conversation import Conversation
from analysis.theses import get_argumentativeness
from utils.lam_results import load_topic_sentences, load_viewpoint_sentences

# switches = ["lex", "pos", "lex_pos"]
# sentence_extractions = ["discriminative", "generative"]
switches = ["lex_pos"]
sentence_extractions = ["discriminative"]

for sentence_extraction in sentence_extractions:
    for switch in switches:
        root_path = f"../results/memo-corpus/lam/{switch}-{sentence_extraction}-topics_first-True-5-2-1000-group1"

        ts_path = "{}/topic_sentences.txt".format(root_path)
        vp_path = "{}/vp_sentences.txt".format(root_path)
        conversation = Conversation("../data/memo-corpus/vtt/merged/group1.vtt")
        conversation.load()

        topic_sentences = load_topic_sentences(ts_path)
        viewpoint_sentences = load_viewpoint_sentences(vp_path)

        sentences_tp = []
        sentences_vp = []

        evals = []

        # topic0_sentences = topic_sentences[0]
        # topic4_sentences = topic_sentences[4]

        # tp sentence selection
        for topic_i_sentences in topic_sentences:
            for sentence in topic_i_sentences[:4]:
                sentences_tp.append(sentence)

        # vp sentence selection
        for topic_i_sentences in viewpoint_sentences:
            for vp_i_sentences in topic_i_sentences:
                for sentence in vp_i_sentences[:2]:
                    sentences_vp.append(sentence)

        tp_scores = []
        vp_scores = []

        for sentence in sentences_tp:
            tp_scores.append(conversation.evaluate_relevance(sentence, ignore_top=True))

        for sentence in sentences_vp:
            vp_scores.append(conversation.evaluate_relevance(sentence, ignore_top=True))

        print("TP scores:")
        print(tp_scores)
        print("VP scores:")
        print(vp_scores)

        print("TP mean score: {}".format(np.mean(tp_scores)))
        print("VP mean score: {}".format(np.mean(vp_scores)))

        tp_args = [get_argumentativeness(thesis) for thesis in tqdm(sentences_tp)]
        vp_args = [get_argumentativeness(thesis) for thesis in tqdm(sentences_vp)]

        print("TP mean argumentativeness: {}".format(np.mean(tp_args)))
        print("VP mean argumentativeness: {}".format(np.mean(vp_args)))

        df = pd.DataFrame()
        df["thesis"] = sentences_vp
        df["evaluation"] = vp_scores
        df["argumentativeness"] = vp_args

        print(np.mean(vp_scores))

        model_name = f"{switch}-{sentence_extraction}"
        df.to_csv(os.path.join("../results/memo-corpus/llm", f"lam-{model_name}-n=2.csv"), index=False)
