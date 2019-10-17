import re
import jellyfish


def sequence_similarity(s1, s2):
    return (len(s1) / (len(s1) + len(s2))) * one_way_sequence_similarity(s1, s2) + \
           (len(s2) / (len(s1) + len(s2))) * one_way_sequence_similarity(s2, s1)


def one_way_sequence_similarity(s1, s2):
    tokens1, tokens2 = get_tokens(s1), get_tokens(s2)
    res = []

    for token1 in tokens1:
        cur_res = [jellyfish.jaro_distance(token1, token2) for token2 in tokens2]
        res.append(max(cur_res))

    return sum(res) / len(res)


def get_tokens(sentence):
    sentence = re.sub("[^a-zA-Z]", " ", sentence.lower())
    sentence = re.sub("\s+", " ", sentence)
    return sentence.split()
