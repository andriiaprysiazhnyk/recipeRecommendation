import os
import jellyfish
import numpy as np
import pandas as pd
from data_merging.embedding_metric import embedding_similarity


def merge(name_metric, is_merged, get_best):
    ingredients = pd.read_csv("../cleaned_data/ingredient.csv", sep=";")
    abbrev = pd.read_csv("../cleaned_data/abbrev.csv", sep=";")

    ingredients["ndb"] = ingredients.apply(lambda x: get_ndb(x, abbrev, name_metric, is_merged, get_best), axis=1)
    merged = ingredients.merge(abbrev, on="ndb", how="inner")
    print(len(merged))

    save_merged(merged)
    update_ingredients(ingredients, merged)


def get_ndb(ingredient, abbrev, name_metric, is_merged, get_best):
    candidates = get_best(abbrev["Shrt_Desc"].apply(lambda x: name_metric(x, ingredient["base"])))

    if not is_merged(candidates):
        return np.NaN

    abbrev = abbrev.loc[candidates.index]
    units = ingredient["unit"].split() if len(ingredient["qty"].split()) > 1 else [ingredient["unit"]]
    return abbrev["ndb"].loc[
        abbrev.apply(
            lambda x: max_distance(list(filter(lambda y: type(y) == str, [x["GmWt_Desc1"], x["GmWt_Desc1"]])), units),
            axis=1).idxmax()]


def max_distance(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 0

    return max(1 - jellyfish.levenshtein_distance(e1, e2) / max(len(e1), len(e2)) for e2 in set2 for e1 in set1)


def save_merged(merged_df):
    merged_file = os.path.join(os.getcwd(), "../", "cleaned_data", "ingredient_abbrev.csv")

    if os.path.exists(merged_file):
        merged_df = pd.concat([pd.read_csv(merged_file, sep=";"), merged_df])

    merged_df.to_csv(merged_file, sep=";", index=False)


def update_ingredients(ingredients, merged_df):
    ingredients = ingredients[ingredients["ingredient_id"].apply(lambda x: x not in list(merged_df["ingredient_id"]))]
    ingredients.drop(columns="ndb", axis=1, inplace=True)
    ingredients.to_csv(os.path.join(os.getcwd(), "../", "cleaned_data", "ingredient.csv"), sep=";", index=False)


if __name__ == "__main__":
    merge(lambda x, y: 1 - jellyfish.levenshtein_distance(x, y) / max(len(x), len(y)), lambda x: max(x) >= 0.75,
          lambda s: s[max(s) - s < 0.01])
    merge(jellyfish.jaro_distance, lambda x: max(x) >= 0.8, lambda s: s[max(s) - s < 0.01])
    merge(embedding_similarity, lambda x: max(x) >= 0.75, lambda s: s[max(s) - s < 0.01])
