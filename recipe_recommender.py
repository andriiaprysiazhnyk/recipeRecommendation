import jellyfish
import pandas as pd


def recommend_recipe(product_pref, chemical_plus, chemical_minus):
    recipes = pd.read_csv("cleaned_data/mapped_recipes.csv", sep=";")
    scores = recipes.apply(
        lambda x: score_products(product_pref, x["ingredients"].split("|")) + score_chemicals(x, chemical_plus, chemical_minus, recipes),
        axis=1)
    return recipes.loc[scores.nlargest(1).index]


def score_products(product_pref, products):
    res = 0

    for name in product_pref:
        scores = list(map(lambda x: jellyfish.jaro_distance(x, name), products))
        if not product_pref[name]:
            scores = list(map(lambda x: 1 - x, scores))

        res += sum(scores)

    return res


def score_chemicals(recipe, chemical_plus, chemical_minus, recipes):
    res = 0

    for element in chemical_plus:
        min_v, max_v = min(recipes[element]), max(recipes[element])
        res += (recipe[element] - min_v) / (max_v - min_v)

    for element in chemical_minus:
        min_v, max_v = min(recipes[element]), max(recipes[element])
        res += 1 - (recipe[element] - min_v) / (max_v - min_v)

    return res


print(recommend_recipe({"egg": False, "fish": True}, {"Calcium": 10}, {}))
