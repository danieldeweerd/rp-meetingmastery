import os.path

import numpy as np
import pandas as pd

from analysis.conversation import Conversation

path = "../data/ami-corpus/vtt/merged/ES2000.vtt"
model_name = "text-davinci-003"
theses_path = "../results/ami-corpus/llm/claude-v1-n=1.csv"
classif_root_path = "../data/ami-corpus/vtt/merged/disagreements/"
theses = pd.read_csv(theses_path)["thesis"].tolist()

conversation = Conversation(path)
conversation.load(merge_consecutive=True)
manual = False
accs = []
pas = []
pds = []
ras = []
rds = []
i = 0
for thesis in theses:
    if not os.path.exists(classif_root_path + thesis.replace(" ", "_") + ".txt"):
        continue
    classif = eval(open(classif_root_path + thesis.replace(" ", "_") + ".txt").read())
    y_true = classif

    if manual:
        conversation.classify_agreement(thesis)
        y_pred = input("Enter predictions for thesis: " + thesis + "\n")
    else:
        y_pred = conversation.classify_agreement(thesis)

    if y_pred[0] == "\'":
        y_pred = y_pred[1:-1]
    y_pred = eval(y_pred)

    while len(y_pred) < len(y_true):
        y_pred.append(0)

    while len(y_pred) > len(y_true):
        y_pred.pop()

    y_true, y_pred = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    # print(len(y_true), len(y_pred))

    accuracy = np.sum(y_true == y_pred) / len(y_true)

    # print(thesis)
    # print("Accuracy:", accuracy)
    # print(accuracy)
    accs.append(accuracy)
    positive_precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    pas.append(positive_precision)
    positive_recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    ras.append(positive_recall)

    negative_precision = np.sum((y_true == -1) & (y_pred == -1)) / np.sum(y_pred == -1)
    pds.append(negative_precision)
    negative_recall = np.sum((y_true == -1) & (y_pred == -1)) / np.sum(y_true == -1)
    rds.append(negative_recall)

    print(accuracy, positive_precision, positive_recall, negative_precision, negative_recall)

    # print("Positive precision:", positive_precision)
    # print(positive_precision)
    # print("Positive rec/all:", positive_recall)
    # print(positive_recall)
    # print("Negative precision:", negative_precision)
    # print(negative_precision)

    # print("Negative recall:", negative_recall)
    # print(negative_recall)
    print("-----")
    # print(y_true)
    # print(y_pred)

pas = np.array(pas)
ras = np.array(ras)
pds = np.array(pds)
rds = np.array(rds)

pas = pas[~np.isnan(pas)]
ras = ras[~np.isnan(ras)]
pds = pds[~np.isnan(pds)]
rds = rds[~np.isnan(rds)]


print("Accuracy:", np.mean(accs))
print("Positive precision:", np.mean(pas))
print("Positive recall:", np.mean(ras))
print("Negative precision:", np.mean(pds))
print("Negative recall:", np.mean(rds))