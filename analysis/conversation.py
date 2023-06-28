import re
import time

import numpy as np
import pyperclip
from scipy.signal import find_peaks
from tqdm import tqdm

from analysis.ai_utils import get_number_of_tokens, get_minilm_embedding_batch, get_embedding, get_minilm_embedding
from analysis.inference import query


class Expression:

    def __init__(self, participant_name, participant_id, phrase, t_0, t_1):
        self.participant_name = participant_name
        self.participant_id = participant_id
        self.phrase = phrase
        self.t_0 = t_0
        self.t_1 = t_1
        self.as_string = f"{self.participant_id}: {self.phrase}"
        self.n_tokens = get_number_of_tokens(self.as_string)

    def __str__(self):
        return f"[{self.participant_name}, {self.participant_id}, {self.phrase}, {self.t_0}, {self.t_1}]"


class Conversation:

    def __init__(self, path):
        self.path = path
        self.participant_names = []
        self.participant_ids = []
        self.expressions = []
        self.expression_embeddings = []
        self.embeddings = []
        self.length = 0

    def load(self, merge_consecutive=True, prettify_dialogue=False, compute_embeddings=True):
        """
        Loads the transcript file and parses it into a list of expressions.
        :param compute_embeddings:
        :param merge_consecutive: whether to merge consecutive expressions from the same participant
        :param prettify_dialogue: whether to add interpunction and capitalization to the expressions with gpt-3.5-turbo
        """
        #####################
        # Basic setup
        #####################
        transcript_file = open(self.path, "r")
        lines = transcript_file.readlines()
        transcript_file.close()

        # The merging algorithm only saves a phrase if the next phrase is from a different participant,
        # this hacky fix ensures that the last 'real' phrase is saved
        lines.append("inf --> inf")
        lines.append("[AlexJanPieter (420)] This will never be saved")

        timestamps = filter(lambda line: "-->" in line, lines)
        users_phrases = filter(lambda line: "[" in line, lines)

        current_t0 = None
        previous_participant_name = None
        previous_participant_id = None
        previous_t1 = None
        current_phrase = ""

        #####################
        # Parse expressions
        #####################
        for timestamp, user_phrase in zip(timestamps, users_phrases):
            splits = timestamp.split(" --> ")
            t_0, t_1 = splits[0], splits[1]
            user_phrase = user_phrase.strip()

            # pattern = r'\[(.+)\s\((\d+)\)\]\s(.+)'
            pattern = r'\[(.+)\]\s(.+)'
            match = re.match(pattern, user_phrase)

            id_present = False
            #
            # if not match:
            #     pattern = r"\[(.+)\](.+)"
            #     match = re.match(pattern, user_phrase)
            #     id_present = False

            if not match:
                # print("No match for phrase:|{}| in file {}".format(user_phrase, self.path))
                # print("Continuing...")
                continue

            participant_name = match.group(1)
            participant_id = int(match.group(2)) if id_present else None
            phrase = match.group(3) if id_present else match.group(2)

            if participant_name not in self.participant_names:
                self.participant_names.append(participant_name)

            if not id_present:
                participant_id = self.participant_names.index(participant_name)

            if participant_id not in self.participant_ids:
                self.participant_ids.append(participant_id)

            if merge_consecutive:
                if previous_participant_id is None:
                    current_phrase = phrase
                    current_t0 = t_0
                elif previous_participant_id == participant_id:
                    current_phrase += " " + phrase
                elif previous_participant_id is not None:
                    if prettify_dialogue:
                        prompt = "Add proper interpunction and capitalization. Output in the following format: [PARAGRAPH]: {}".format(
                            current_phrase)
                        current_phrase = query(prompt, model_name="gpt-3.5-turbo")[13:]

                    self.expressions.append(
                        Expression(previous_participant_name, previous_participant_id, current_phrase, current_t0,
                                   previous_t1))
                    current_phrase = phrase
                    current_t0 = t_0
            else:
                self.expressions.append(Expression(participant_name, participant_id, phrase, t_0, t_1))

            previous_participant_name = participant_name
            previous_participant_id = participant_id
            previous_t1 = t_1

        #####################
        # Compute embeddings
        #####################

        if compute_embeddings:
            phrases = [expression.phrase for expression in self.expressions]
            embeddings = get_minilm_embedding_batch(phrases)
            self.embeddings = np.array(embeddings)
            print("Stored embeddings as {}x{} matrix".format(self.embeddings.shape[0], self.embeddings.shape[1]))

    def split_chunks(self, n_tokens_per_chunk=2500):
        """
        Splits the expressions into chunks of n_tokens_per_chunk tokens.
        :param n_tokens_per_chunk: the number of tokens per chunk
        :return: a list of chunks, where each chunk is a list of expressions
        """
        chunks = []
        tokens_in_chunk = 0
        chunks.append([])
        i = 0
        for expression in self.expressions:
            n_tokens_expr = expression.n_tokens
            if tokens_in_chunk + n_tokens_expr < n_tokens_per_chunk:
                chunks[i].append(expression)
                tokens_in_chunk += n_tokens_expr
            else:
                chunks.append([])
                i += 1
                chunks[i].append(expression)
                tokens_in_chunk = n_tokens_expr

        return chunks

    def extract_theses(self, model_name="gpt-3.5-turbo", n=1, n_tokens_per_chunk=1500, max_length=1000,
                       truncate_thesis=False, sleep=0, save_prompts=False):
        chunks = self.split_chunks(n_tokens_per_chunk=n_tokens_per_chunk)
        prompt = "From the following dialogue, extract an argumentative thesis with which some dialogue " \
                 "participants agree and others disagree.You should formulate the thesis in a single " \
                 "sentence. It should be very concise, affirmatively formulated, and contain a clear claim." \
                 " It should not contain nuance or caveats. Again, it should be terse. Output only the" \
                 " thesis and nothing else.\n Dialogue:\n{} Thesis:"

        theses = []
        prompts = []
        for chunk in tqdm(chunks, desc="Extracting theses per chunk"):
            dialogue = ""
            for expression in chunk:
                dialogue += expression.as_string + "\n"

            for _ in range(n):
                thesis = query(prompt.format(dialogue), model_name=model_name, max_length=max_length)
                prompts.append(prompt.format(dialogue))
                if truncate_thesis:
                    thesis = thesis.split(".")[0] + "."
                # thesis.replace("\n", " ")
                theses.append(thesis)
                time.sleep(sleep)

        if save_prompts:
            with open("prompts.txt", "w") as f:
                for prompt in prompts:
                    f.write(prompt + "\n")

        return theses

    def extract_theses_manually(self, n=1, n_tokens_per_chunk=1500,
                                truncate_thesis=False, sleep=0, save_prompts=False):
        chunks = self.split_chunks(n_tokens_per_chunk=n_tokens_per_chunk)
        prompt = "From the following dialogue, extract an argumentative thesis with which some dialogue " \
                 "participants agree and others disagree.You should formulate the thesis in a single " \
                 "sentence. It should be very concise, affirmatively formulated, and contain a clear claim." \
                 " It should not contain nuance or caveats. Again, it should be terse. Output only the" \
                 " thesis and nothing else.\n Dialogue:\n{} Thesis:"

        theses = []
        prompts = []
        for chunk in chunks:
            dialogue = ""
            for expression in chunk:
                dialogue += expression.as_string + "\n"

            for _ in range(n):
                pyperclip.copy(prompt.format(dialogue))
                thesis = input("Enter the retrieved thesis:")
                if thesis == "exit()":
                    return theses

                prompts.append(prompt.format(dialogue))
                if truncate_thesis:
                    thesis = thesis.split(".")[0] + "."
                # thesis.replace("\n", " ")
                theses.append(thesis)
                time.sleep(sleep)

        if save_prompts:
            with open("prompts.txt", "w") as f:
                for prompt in prompts:
                    f.write(prompt + "\n")

        return theses

    def get_relevant_expressions(self, thesis, sliding_window_size=100):
        thesis_embedding = get_minilm_embedding(thesis)
        sim_values = np.dot(self.embeddings, thesis_embedding)
        sim_values = np.convolve(sim_values, np.ones(sliding_window_size) / sliding_window_size, mode='same')
        peaks, _ = find_peaks(sim_values)
        peaks = peaks[np.argsort(sim_values[peaks])][::-1][:2]
        peaks = np.sort(peaks)
        index_0 = peaks[0]
        return self.expressions[index_0 - 20:index_0 + 20]

    def classify_agreement(self, thesis, model_name="gpt-3.5-turbo", manual_mode=False):
        template = "Task: for the given argumentative thesis and dialogue, indicate for every sentence whether it agrees with, " \
                   "disagrees with, or is unrelated to the thesis. Give your response as a Python list per sentences, " \
                   "where 0 means unrelated, 1 means agree, and -1 means disagree. Output only the " \
                   "Python list and nothing else.\n " \
                   "Thesis: {}\n" \
                   "Dialogue:\n"

        prompt = template.format(thesis)
        relevant_expressions = self.get_relevant_expressions(thesis)
        for i, expr in enumerate(relevant_expressions):
            line = "{}. [Speaker {}]: {}".format(i + 1, expr.participant_id, expr.phrase)
            prompt += line + "\n"
        # pyperclip.copy(prompt)
        results = query(prompt, model_name=model_name)
        return results

    def semantic_similarity(self, thesis):
        thesis_embedding = get_embedding(thesis)
        similarities = np.matmul(self.embeddings, thesis_embedding)
        return similarities

    def evaluate_relevance(self, thesis, k=10, ignore_top=False):
        similarities = self.semantic_similarity(thesis)
        similarities = np.sort(similarities)[::-1]
        return np.mean(similarities[1:k + 1]) if ignore_top else np.mean(similarities[:k])

    def get_most_similar(self, phrase, n=-1):
        phrase_embedding = get_embedding(phrase)
        similarities = np.matmul(self.embeddings, phrase_embedding)
        indices = np.argsort(similarities)[::-1]
        return [self.expressions[i].phrase for i in indices[:n]]

    def export_txt(self, path="export.txt", include_timestamps=True):
        with open(path, "w") as export_file:
            export_file.write("WEBVTT" + "\n")
            export_file.write("\n")

            for expression in self.expressions:
                if include_timestamps:
                    export_file.write(expression.t_0 + " --> " + expression.t_1)
                line = "[{} ({})] {} \n".format(expression.participant_name, expression.participant_id,
                                                expression.phrase)
                export_file.write(line)

    def export_tsv(self, path="export.tsv"):
        with open(path, "w") as export_file:
            for expression in self.expressions:
                line = "{}\t{}\t{}\t{}\t{}"

                line = line.format(expression.t_0, "PLACEHOLDER-1", "PLACEHOLDER-2", "PLACEHOLDER-3", expression.phrase)
                export_file.write(line)
                export_file.write("\n")
