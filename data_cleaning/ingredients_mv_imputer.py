import jellyfish
import pandas as pd


def impute_ingredients_mv(ingredients):
    ingredients_clean = ingredients.dropna(subset=["qty", "unit"])
    ingredients_clean = ingredients_clean[ingredients_clean["qty"].apply(lambda x: len(x.split()) == 1)]
    ingredients["unit"] = ingredients.apply(
        lambda x: impute_unit(x, ingredients_clean) if pd.isna(x["unit"]) else x["unit"], axis=1)
    ingredients["qty"] = ingredients.apply(
        lambda x: impute_qty(x, ingredients_clean) if pd.isna(x["qty"]) else x["qty"], axis=1)


def impute_unit(instance, ingredients):
    similar_ingredients = get_similar(instance, ingredients,
                                      lambda x, y: jellyfish.damerau_levenshtein_distance(x["base"], y["base"]))

    return similar_ingredients.unit.mode()


def impute_qty(instance, ingredients):
    similar_ingredients = get_similar(instance, ingredients,
                                      lambda x, y: jellyfish.damerau_levenshtein_distance(x["base"], y[
                                          "base"]) + jellyfish.damerau_levenshtein_distance(x["unit"],
                                                                                            y["unit"]))

    return similar_ingredients.qty.apply(float).mean()


def get_similar(instance, ingredients, distance):
    n_neighbors = 100
    scores = ingredients.apply(lambda x: distance(x, instance), axis=1)
    return ingredients.loc[scores.nsmallest(n_neighbors).index]
