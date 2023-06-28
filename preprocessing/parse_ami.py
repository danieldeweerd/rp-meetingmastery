import json
import os


def frmt(text):
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" ?", "?")
    text = text.replace("<vocalsound>", "")
    text = text.replace("<disfmarker>", "")
    text = text.replace("  ", " ")
    return text.strip()


path = "data/ami-corpus/dialogueActs/"
outpath = "data/ami-corpus/vtt/"

if not os.path.exists(outpath):
    os.makedirs(outpath)

fnames = os.listdir(path)

for fname in fnames:
    filepath = os.path.join(path, fname)
    with open(filepath, "r") as f:
        data = json.load(f)

    speakers, phrases, timestamps = [], [], []

    speaker_names = []

    for d in data:
        phrase = d["text"]
        speaker = d["speaker"]
        t_0 = d["starttime"]
        t_1 = d["endtime"]
        timestamp = f"{t_0} --> {t_1}"

        speakers.append(speaker)
        phrases.append(phrase)
        timestamps.append(timestamp)

        if speaker not in speaker_names:
            speaker_names.append(speaker)

    with open(os.path.join(outpath, fname.split(".")[0] + ".vtt"), "w") as f:
        f.write("WEBVTT\n\n")
        for speaker, phrase, timestamp in zip(speakers, phrases, timestamps):
            f.write(f"{timestamp}\n")
            f.write(f"[{speaker} ({speaker_names.index(speaker)})] {frmt(phrase)}\n")
            f.write("\n")
