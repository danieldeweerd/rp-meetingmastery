from conversation import Conversation
from tqdm import tqdm

for group in tqdm(range(1, 16)):
    if group == 8:
        continue
    for session in range(1, 4):
        path = "data/memo-corpus/vtt/group{}_session{}_video.vtt".format(group, session)
        conversation = Conversation(path)
        conversation.load(merge_consecutive=True, prettify_dialogue=False, compute_embeddings=False)
        conversation.export_tsv("data/memo-corpus/tsv/group{}_session{}.tsv".format(group, session))
    # path = "memo-corpus/group{}_session{}_video.vtt".format(group, 1)

# path = "data/group5_session2_video.vtt"
# conversation = Conversation(path)
# conversation.load(merge_consecutive=True, prettify_dialogue=False)
#
# conversation.export_tsv("prettified.tsv")