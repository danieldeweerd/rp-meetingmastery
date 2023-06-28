from tqdm import tqdm

import os

from ai_utils import get_number_of_tokens

folder = "data/ami-corpus/vtt/"

l = 0

for fname in tqdm(os.listdir(folder)):
    txt = open(folder + fname).read()
    l += get_number_of_tokens(txt)

l /= 1000

ada_price = 0.0004
babbage_price = 0.0005
curie_price = 0.002
davinci_price = 0.02
turbo_price = 0.002
gpt4_price = 0.03

print("ada: €{}".format(l * ada_price))
print("babbage: €{}".format(l * babbage_price))
print("curie: €{}".format(l * curie_price))
print("davinci: €{}".format(l * davinci_price))
print("turbo: €{}".format(l * turbo_price))
print("gpt4: €{}".format(l * gpt4_price))
