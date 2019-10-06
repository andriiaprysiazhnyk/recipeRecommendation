import jellyfish


def recipe_similarity(recipe1, recipe2):
    return (tags_similarity(recipe1, recipe2) + ingredients_similarity(recipe1, recipe2)
            + chemicals_similarity(recipe1, recipe2)) / 3


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
