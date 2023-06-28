from analysis.conversation import Conversation

path = "data/ami-corpus/vtt/merged/ES2000.vtt"

conversation = Conversation(path)
conversation.load(compute_embeddings=False, merge_consecutive=False)

print(len(conversation.expressions))
sum_words = 0
sum_chars = 0

for expr in conversation.expressions:
    sum_words += len(expr.phrase.split(" "))
    sum_chars += len(expr.phrase)

print("Average words per expression: {}".format(sum_words / len(conversation.expressions)))
print("Average chars per expression: {}".format(sum_chars / len(conversation.expressions)))

print("Total words: {}".format(sum_words))
print("Total chars: {}".format(sum_chars))
