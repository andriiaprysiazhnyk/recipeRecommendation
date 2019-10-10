import pandas as pd
from recipe_comparator import ingredient_similarity
from utility_based_recommendation.utility_based_recommender import recommend_recipes


def get_best_recipe(product_pref, chemical_pref):
    recipes = pd.read_csv("cleaned_data/mapped_recipes.csv", sep=";")
    scores = recipes.apply(
        lambda x: score_products(product_pref, x["ingredients"].split("|")) + score_chemicals(x, chemical_pref, recipes),
        axis=1)

    best_recipe = recipes.loc[scores.nlargest(1).index[0]]
    return best_recipe, recommend_recipes(best_recipe)


def score_products(product_pref, products):
    res = 0

    for name in product_pref:
        scores = list(map(lambda x: ingredient_similarity(name, x), products))
        res += max(scores) if product_pref[name] else min(scores)

    return res


def score_chemicals(recipe, chemical_pref, recipes):
    res = 0

    for element in chemical_pref:
        max_deviation = max(abs(recipes[element] - chemical_pref[element][0]))
        zero_value, one_value = chemical_pref[element][0] - max_deviation, chemical_pref[element][0] + max_deviation
        cur_res = (recipe[element] - zero_value) / (one_value - zero_value)

        res += cur_res if chemical_pref[element][1] == "+" else (1 - cur_res)

    return res
