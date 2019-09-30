import jellyfish
import pandas as pd
import pickle


def generate_most_similar_recipes():
    recipes = pd.read_csv("cleaned_data/mapped_recipes.csv", sep=";")

    non_chemical_columns = ["recipe_id", "title", "author", "url", "tags", "servings", "ingredients"]
    chemicals_columns = list(recipes.columns)
    for non_chemical in non_chemical_columns:
        chemicals_columns.remove(non_chemical)

    chemicals = recipes.drop(non_chemical_columns, axis=1)
    recipes[chemicals_columns] = (chemicals - chemicals.min()) / (chemicals.max() - chemicals.min())

    most_similar = {}
    for _, recipe in recipes.iterrows():
        scores = recipes.apply(lambda x: 0 if x["recipe_id"] == recipe["recipe_id"] else recipe_similarity(recipe, x),
                               axis=1)
        most_similar[recipe["recipe_id"]] = list(recipes["recipe_id"].loc[scores.nlargest(3).index])

    pickle.dump(most_similar, open("similar_recipes.pickle", "wb"))


def recipe_similarity(recipe1, recipe2):
    return tags_similarity(recipe1, recipe2) + ingredients_similarity(recipe1, recipe2) + chemicals_similarity(recipe1,
                                                                                                               recipe2)


def tags_similarity(recipe1, recipe2):
    return seq_similarity(recipe1["tags"].split("|"), recipe2["tags"].split("|"))


def ingredients_similarity(recipe1, recipe2):
    return seq_similarity(recipe1["ingredients"].split("|"), recipe2["ingredients"].split("|"))


def seq_similarity(seq1, seq2):
    if len(seq1) == 0: return 0

    res = 0

    for element1 in seq1:
        res += max((1 - jellyfish.levenshtein_distance(element1, element2) / max(len(element1), len(element2)))
                   for element2 in seq2) if element1 else 0

    return res / len(seq1)


def chemicals_similarity(recipe1, recipe2):
    non_chemical_columns = ["recipe_id", "title", "author", "url", "tags", "servings", "ingredients"]
    recipe1 = recipe1.drop(labels=non_chemical_columns)
    recipe2 = recipe2.drop(labels=non_chemical_columns)

    return 1 - sum((recipe1 - recipe2) ** 2) / len(recipe1)


generate_most_similar_recipes()
