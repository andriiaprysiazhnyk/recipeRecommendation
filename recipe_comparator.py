import os
import jellyfish
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

path_to_model = ("" if os.getcwd().endswith(
    "recipeRecommendation2") else "../") + "embedding/GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)


def recipe_similarity(recipe1, recipe2):
    return .25 * tags_similarity(recipe1, recipe2) + .5 * ingredients_similarity(recipe1, recipe2) + \
           .25 * chemicals_similarity(recipe1, recipe2)


def tags_similarity(recipe1, recipe2):
    tags1, tags2 = recipe1["tags"].split("|"), recipe2["tags"].split("|")
    tags1, tags2 = list(filter(lambda x: x, tags1)), list(filter(lambda x: x, tags2))

    return seq_similarity(tags1, tags2,
                          lambda x, y: 1 - jellyfish.levenshtein_distance(x, y) / max(len(x), len(y)))


def ingredients_similarity(recipe1, recipe2):
    return seq_similarity(recipe1["ingredients"].split("|"), recipe2["ingredients"].split("|"), ingredient_similarity)


def ingredient_similarity(ingredient1, ingredient2):
    return cosine_similarity(seq2vec(ingredient1), seq2vec(ingredient2))[0, 0]


def seq2vec(seq):
    seq = seq.split()

    for i in range(len(seq)):
        seq[i] = "".join([c for c in seq[i] if c not in string.punctuation])
    seq = list(filter(lambda x: x in model.vocab, seq))

    return np.mean([model[token] for token in seq], axis=0).reshape((1, -1))


def seq_similarity(seq1, seq2, similarity_func):
    def one_way_seq_similarity(sequence1, sequence2):
        res = 0

        for element1 in sequence1:
            res += max(similarity_func(element1, element2) for element2 in sequence2)

        return res / len(sequence1)

    if len(seq1) == 0 or len(seq2) == 0:
        return 0

    c1, c2 = len(seq1) / (len(seq1) + len(seq2)), len(seq2) / (len(seq1) + len(seq2))
    return c1 * one_way_seq_similarity(seq1, seq2) + c2 * one_way_seq_similarity(seq2, seq1)


def chemicals_similarity(recipe1, recipe2):
    non_chemical_columns = ["recipe_id", "title", "author", "url", "tags", "ingredients"]
    recipe1 = recipe1.drop(labels=non_chemical_columns)
    recipe2 = recipe2.drop(labels=non_chemical_columns)

    return 1 - sum((recipe1 - recipe2) ** 2) / len(recipe1)


def normalize_recipes(recipes):
    non_chemical_columns = ["recipe_id", "title", "author", "url", "tags", "ingredients"]
    chemicals_columns = list(recipes.columns)
    for non_chemical in non_chemical_columns:
        chemicals_columns.remove(non_chemical)

    chemicals = recipes.drop(non_chemical_columns, axis=1)
    recipes[chemicals_columns] = (chemicals - chemicals.min()) / (chemicals.max() - chemicals.min())
