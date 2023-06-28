# load all lines in "topic_sentences.txt"

topic_filepath = "../results/memo-corpus/lam/lex_pos-discriminative-topics_first-True-5-2-1000-group1/topic_sentences.txt"
vp_filepath = "../results/memo-corpus/lam/lex_pos-discriminative-topics_first-True-5-2-1000-group1/vp_sentences.txt"


def load_topic_sentences(filepath, n_topics=5):
    topic_sentences = []

    for i in range(n_topics):
        topic_sentences.append([])

    curr_topic = -1

    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == f"Topic: {curr_topic + 1}":
                curr_topic += 1
            if "\t" in line:
                topic_sentences[curr_topic].append(line.split("\t")[1].strip())

    return topic_sentences


def load_viewpoint_sentences(filepath, n_topics=5, n_viewpoints=2):
    viewpoint_sentences = []

    for i in range(n_topics):
        viewpoints = []
        for j in range(n_viewpoints):
            viewpoints.append([])

        viewpoint_sentences.append(viewpoints)

    curr_topic = -1
    curr_vp = -1

    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if f"Topic: {curr_topic + 1}" in line:
                curr_topic += 1
            if f"Viewpoint: {curr_vp + 1}" in line:
                curr_vp += 1
            if f"Viewpoint: {curr_vp - 1}" in line:
                curr_vp -= 1
            if "\t" in line:
                viewpoint_sentences[curr_topic][curr_vp].append(line.split("\t")[1].strip())

    return viewpoint_sentences


# tps = load_topic_sentences(topic_filepath)
# vps = load_viewpoint_sentences(vp_filepath)
#
# # for i in range(5):
# #     for j in range(2):
# #         print(f"Topic {i} viewpoint {j}:")
# #         for sentence in vps[i][j]:
# #             print(sentence)
# #         print()
