from analysis.conversation import Conversation

conversation = Conversation("../data/ami-corpus/vtt/merged/ES2000.vtt")
conversation.load()
# conversation.export_tsv("../data/ami-corpus/tsv/merged/ES2000.tsv")

expressions = conversation.expressions
n_words = sum([len(expression.phrase.split(" ")) for expression in expressions])
n_tokens = sum([expression.n_tokens for expression in expressions])

print("Number of expressions: {}".format(len(expressions)))
print("Number of words: {}".format(n_words))
print("Number of tokens: {}".format(n_tokens))

# conversation.extract_theses(n=1, n_tokens_per_chunk=1500, max_length=300, model_name="text-ada-001", sleep=5)
