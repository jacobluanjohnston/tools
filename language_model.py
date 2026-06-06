"""
This file was originally live coded by UCSC professor Alexander Rudnick.
"""

from collections import Counter
from collections import defaultdict

import math
import random

START = "<START>"
END = "<END>"

def load_counts(corpus_fn):
    contexts_to_counts = defaultdict(Counter)
    with open(corpus_fn) as infile:
        for line in infile:
            context = (START, START)
            line = line.strip().lower()
            tokens = line.split() + [END]

            for token in tokens:
                ngram = context + (token,)
                # print("here's an ngram!", ngram)
                contexts_to_counts[context][token] += 1
                context = context[1:] + (token,)
    return contexts_to_counts

def counts_to_probabilities(contexts_to_counts):
    contexts_to_probs = defaultdict(lambda: defaultdict(float))
    for context, count in contexts_to_counts.keys():

        # considering a specific context

        probs = defaultdict(float)
        # probability of a word in a context is
        # c(word in context) / c(that context)

        total_count = sum(contexts_to_counts[context].values())
        if not total_count: continue
        for wordtype in contexts_to_counts[context].keys():
            count = contexts_to_counts[context][wordtype]
            contexts_to_probs[context][wordtype] = count / total_count

    return contexts_to_probs

def score(contexts_to_probs, sentence):
    """Returns the bits of surprise for this sequence according to the model."""
    ##
    total_bits = 0
    context = (START, START)
    for token in sentence + [END]:
        prob = contexts_to_probs[context][token]
        if (not prob):
            prob = 1e-7
            # print("warning!! zero probability!")
        bits = -math.log2(prob)
        total_bits += bits
        context = context[1:] + (token,)
    return total_bits

def sample_sentence(contexts_to_probs):
    context = (START, START)
    # help(random.choices)
    # random.choices(["hello", "goodbye", "avocado"], [0.2, 0.2, 0.6])
    output = []
    while True:
        possible_next_words = list(contexts_to_probs[context].keys())
        next_word_dist = list(contexts_to_probs[context].values())
        next_word = random.choices(possible_next_words, weights=next_word_dist)[0]

        if next_word == END:
            return output
        output.append(next_word)
        context = context[1:] + (next_word,)

def main():
    contexts_to_counts = load_counts("data/language_model/corpus_sentences.txt")
    # print(contexts_to_counts[(START, START)])
    contexts_to_probs = counts_to_probabilities(contexts_to_counts)

    print(contexts_to_probs[(START, "the")])

    ## scoring sentences works like this....
    ## PREPROCESSING MATTERS!
    sentence = "Call me Ishmael .".lower().split()
    print("score for a real sentence", score(contexts_to_probs, sentence))

    sentence = "this sentence avocado banana has never existaroonied .".lower().split()
    print("score for a fake sentence", score(contexts_to_probs, sentence))
    # And that got zero probability!

    for _ in range(10):
        sent = sample_sentence(contexts_to_probs)
        print(sent)

if __name__ == "__main__": main()