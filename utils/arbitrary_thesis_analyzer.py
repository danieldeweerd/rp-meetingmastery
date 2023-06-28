from analysis.conversation import Conversation
from analysis.inference import query

thesis = "The remote control should have voice recognition."
n = 10

conversation = Conversation("../data/ami-corpus/vtt/merged/ES2000.vtt")
conversation.load()

sims = conversation.get_most_similar(thesis, n=10)

template = "Thesis: {}\nTask: For every line in the following transcript, determine whether it agrees, disagrees" \
           " or is neutral with respect to the thesis. Give your output as a JSON with the line numbers as keys and" \
           "a 2-tuple as value, the first value of which is one of [\"agree\", \"disagree\", \"neutral\"] and the" \
           "second value is an explanation.\nLine:\n".format(thesis)
for i in range(n):
    template += "{}. {}\n".format(i + 1, sims[i])

# res = query(template, "dolly-12b", max_length=5000)

print(template)
# print(res)

