import pandas as pd
import pickle


from recipe_comparator import recipe_similarity


def recommend_recipes(recipe):
    recipes = pd.read_csv("cleaned_data/mapped_recipes.csv", sep=";")
    similar_recipes_dict = pickle.load(open("utility_based_recommendation/similar_recipes.pickle", "rb"))
    similar_recipes_id = similar_recipes_dict[recipe["recipe_id"]]

    return recipes[recipes["recipe_id"].apply(lambda x: x in similar_recipes_id)]


def generate_most_similar_recipes():
    recipes = pd.read_csv("../cleaned_data/mapped_recipes.csv", sep=";")

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


if __name__ == "__main__":
    generate_most_similar_recipes()
