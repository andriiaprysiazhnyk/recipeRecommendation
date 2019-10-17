import numpy as np
import pandas as pd

from recipe_comparator import recipe_similarity, normalize_recipes


def generate_recipes_similarity_matrix():
    recipes = pd.read_csv("../cleaned_data/mapped_recipes.csv", sep=";")
    similarity_matrix = np.empty((len(recipes), len(recipes)))

    normalize_recipes(recipes)

    for i in range(len(recipes)):

        if i % 50 == 0:
            print("{} element are processed".format(i))

        for j in range(i, len(recipes)):
            similarity_matrix[i, j] = recipe_similarity(recipes.iloc[i], recipes.iloc[j]) if i != j else 1
            similarity_matrix[j, i] = similarity_matrix[i, j]

    pd.DataFrame(similarity_matrix).to_csv("recipes_similarities.csv", index=False)


if __name__ == "__main__":
    generate_recipes_similarity_matrix()
