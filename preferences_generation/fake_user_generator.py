import numpy as np
import pandas as pd
from recipe_recommender import score_chemicals, score_products


def get_scores(recipes, liked_products, disliked_products, chemical_pref):
    product_pref = dict(list(zip(liked_products, [True] * len(liked_products))) + list(
        zip(disliked_products, [False] * len(disliked_products))))

    return recipes.apply(
        lambda x: score_products(product_pref, x["ingredients"].split("|")) + score_chemicals(x, chemical_pref,
                                                                                              recipes), axis=1)


def generate_fake_user_preferences():
    recipes = pd.read_csv("../cleaned_data/mapped_recipes.csv", sep=";")

    liked_products = ["broccoli", "carrot", "cabbage", "avocado", "egg", "almonds", "cucumber", "cauliflower", "beans",
                      "lentils", "yogurt", "extra virgin olive oil", "kale", "asparagus"]
    disliked_products = ["meat", "chicken", "beef", "pork", "sugar", "ham", "turkey", "sausage", "rice"]
    chemical_pref = {"Cholestrl": (55, "-"), "Protein": (25, "-"), "Energ_Kcal": (500, "-"), "Vit_K": (40, "-"),
                     "Vit_C": (40, "-")}

    scores = get_scores(recipes, liked_products, disliked_products, chemical_pref)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores = scores.apply(lambda x: float("NaN") if np.random.choice([0, 1], p=[0.6, 0.4]) else 10 * x)
    pd.DataFrame(scores).to_csv("preferences.csv", index=False)


if __name__ == "__main__":
    generate_fake_user_preferences()
