from analysis.conversation import Conversation

conversation = Conversation("../data/ami-corpus/vtt/merged/ES2000.vtt")
conversation.load(compute_embeddings=False)

i = 1
with open("convo-ES2000.txt", "w") as f:
    for expression in conversation.expressions:
        fmt = "{}. {}".format(i, expression.phrase)
        f.write(fmt + "\n")
        i += 1

