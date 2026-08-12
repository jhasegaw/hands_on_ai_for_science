"""
Generate word_vectors_50d.npz -- a small set of pretrained word embeddings
used in the Aug 13 afternoon embeddings session (nearest neighbors and the
vector-arithmetic demo).

Attendees do not need to run this file: the .npz it produces is already in
the repository (41 KB).  Running it downloads the full 66 MB GloVe file,
which is exactly the kind of conference-wifi load the workshop avoids.

Source: GloVe (Global Vectors for Word Representation), Pennington,
Socher & Manning (2014), trained on Wikipedia 2014 + Gigaword 5
(glove-wiki-gigaword-50, via the gensim-data mirror).  GloVe vectors are
released under the Public Domain Dedication and License (PDDL).

The subset keeps ~220 curated words: the classic analogy sets
(king/queen, country/capital, verb tenses, comparatives) plus common
animals, foods, body parts, science terms, colors, and numbers -- enough
for nearest-neighbor exploration without shipping 400,000 words.

Load with:
    z = np.load('word_vectors_50d.npz', allow_pickle=False)
    words, vectors = list(z['words']), z['vectors']   # (n, 50) float32
"""

import gzip
import urllib.request

import numpy as np

URL = ("https://github.com/piskvorky/gensim-data/releases/download/"
       "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz")

WORDS = """
king queen man woman prince princess boy girl father mother brother sister uncle aunt
paris france london england tokyo japan rome italy berlin germany madrid spain moscow russia
athens greece cairo egypt ottawa canada
walking walked swimming swam running ran eating ate speaking spoke
good better best bad worse worst big bigger biggest small smaller smallest fast faster
cat dog mouse rat horse cow pig sheep goat bear fox wolf lion tiger elephant monkey
bird eagle hawk sparrow fish salmon shark whale dolphin snake frog insect bee ant spider
apple banana orange grape lemon fruit vegetable potato tomato corn wheat rice bread cheese
water fire earth air ice steam rain snow wind storm river ocean lake mountain valley forest
doctor nurse hospital patient disease medicine drug cancer virus bacteria infection vaccine
science scientist biology chemistry physics mathematics laboratory experiment theory data
gene protein cell brain neuron blood heart lung liver kidney muscle bone skin
university student professor teacher school book paper research study knowledge
computer machine software program algorithm robot engine wheel tool metal plastic
happy sad angry afraid love hate hope fear joy pain
red blue green yellow black white color light dark
one two three four five six seven eight nine ten hundred thousand million
day night morning evening week month year hour minute second summer winter spring autumn
""".split()


def main():
    print("downloading", URL, "(~66 MB)...")
    urllib.request.urlretrieve(URL, "glove50.gz")

    want = set(WORDS)
    words, vecs = [], []
    with gzip.open("glove50.gz", "rt", encoding="utf8") as f:
        f.readline()  # header line: "400000 50"
        for line in f:
            w, rest = line.split(" ", 1)
            if w in want:
                words.append(w)
                vecs.append(np.fromstring(rest, sep=" "))
                if len(words) == len(want):
                    break

    np.savez_compressed("word_vectors_50d.npz",
                        words=np.array(words),
                        vectors=np.stack(vecs).astype(np.float32))
    print(f"wrote word_vectors_50d.npz: {len(words)} words x 50 dims")


if __name__ == "__main__":
    main()
