import os

from tqdm import tqdm

from conversation import Conversation

path = "data/memo-corpus/vtt"
for file in os.listdir(path):
    if "group12" in file:
        continue

    filepath = os.path.join(path, file)
    conv = Conversation(filepath)
    conv.load(compute_embeddings=False)
    conv.export_tsv("data/memo-corpus/tsv/sessions" + file[:-4] + ".tsv")


# for i in range(1,16):


# all_lines = []
# for file in tqdm(os.listdir("tsvs/")):
#     transcript_file = open("tsvs/" + file, "r")
#     lines = transcript_file.readlines()
#     transcript_file.close()
#     all_lines.extend(lines)
#
# with open("memo-may16.tsv", "w") as f:
#     f.writelines(all_lines)
#
