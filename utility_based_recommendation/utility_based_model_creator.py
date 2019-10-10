import pandas as pd
import pickle


from recipe_comparator import recipe_similarity, normalize_recipes


def generate_most_similar_recipes():
    recipes = pd.read_csv("../cleaned_data/mapped_recipes.csv", sep=";")

    normalize_recipes(recipes)

    most_similar = {}
    for _, recipe in recipes.iterrows():
        scores = recipes.apply(lambda x: 0 if x["recipe_id"] == recipe["recipe_id"] else recipe_similarity(recipe, x),
                               axis=1)
        most_similar[recipe["recipe_id"]] = list(recipes["recipe_id"].loc[scores.nlargest(3).index])

    pickle.dump(most_similar, open("similar_recipes.pickle", "wb"))


if __name__ == "__main__":
    generate_most_similar_recipes()
