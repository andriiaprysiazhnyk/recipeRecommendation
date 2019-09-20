import os
import re
import jellyfish
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    os.path.join(os.getcwd(), "../", "embedding", "GoogleNews-vectors-negative300.bin"), binary=True)


def embedding_similarity(s1, s2):
    return (one_way_embedding_similarity(s1, s2) + one_way_embedding_similarity(s2, s1)) / 2


def one_way_embedding_similarity(s1, s2):
    tokens1, tokens2 = get_tokens(s1), get_tokens(s2)
    in_model = dict((token2, token2 in model.vocab) for token2 in tokens2)
    res = []

    for token1 in tokens1:
        cur_res = [jellyfish.jaro_distance(token1, token2) for token2 in tokens2]
        if token1 in model.vocab:
            cur_res += [model.similarity(token1, token2) for token2 in tokens2 if in_model[token2]]
        res.append(max(cur_res))

    return sum(res) / len(res)


def get_tokens(sentence):
    sentence = re.sub("[^a-zA-Z]", " ", sentence.lower())
    sentence = re.sub("\s+", " ", sentence)
    return sentence.split()
