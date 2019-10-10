import pandas as pd
import pickle


def recommend_recipes(recipe):
    recipes = pd.read_csv("cleaned_data/mapped_recipes.csv", sep=";")
    similar_recipes_dict = pickle.load(open("utility_based_recommendation/similar_recipes.pickle", "rb"))
    similar_recipes_id = similar_recipes_dict[recipe["recipe_id"]]

    return recipes[recipes["recipe_id"].apply(lambda x: x in similar_recipes_id)]
