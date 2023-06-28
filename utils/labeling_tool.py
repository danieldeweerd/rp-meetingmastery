

import json
import os
from analysis.conversation import Conversation


thesis = "The policies implemented during the pandemic were necessary and had their part to play, even if some were difficult or seemed unfair, and we should have gone into lockdown quicker and closed international borders to prevent the spread of the virus."
folder = "../data/processed/"
file_name = "group5_session2"
path = os.path.join(folder, file_name + "_video.vtt")
json_object = []
labels = []

conversation = Conversation(path)
conversation.load()

length = len(conversation.expressions)

for i, expression in enumerate(conversation.expressions):
    os.system("cls")
    print("({}/{})".format(i + 1, length))
    print("Thesis: {}".format(thesis))
    phrase = expression.phrase
    c = input("Phrase: {} \n".format(phrase))
    if c == "a":
        labels.append(1)
    elif c == "d":
        labels.append(-1)
    else:
        labels.append(0)
    
    

obj = {}
obj["thesis"] = thesis
obj["labels"] = labels
json_object.append(obj)

with open(f"data/processed/{file_name}.json", "w") as f:
    json.dump(json_object, f)