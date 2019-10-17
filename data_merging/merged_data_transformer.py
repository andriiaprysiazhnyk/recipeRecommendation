import os
import jellyfish
import pandas as pd


def transform_merged_data():
    merged_file = os.path.join(os.getcwd(), "../", "cleaned_data", "ingredient_abbrev.csv")
    merged_ingredients = pd.read_csv(merged_file, sep=";")

    unnecessary_columns = ["unit", "qty", "GmWt_1", "GmWt_Desc1", "GmWt_2", "GmWt_Desc2", "Refuse_Pct", "ndb"]
    non_chemical_columns = ["ingredient_id", "base", "description", "Shrt_Desc"]

    merged_ingredients["grams"] = merged_ingredients.apply(lambda x: get_grams(x), axis=1)
    merged_ingredients.drop(columns=unnecessary_columns, axis=1, inplace=True)

    for column in merged_ingredients.columns:
        if column not in non_chemical_columns:
            merged_ingredients[column] = merged_ingredients[column] * merged_ingredients["grams"] / 100
            merged_ingredients[column] /= merged_ingredients["servings"]

    merged_ingredients.drop(columns="grams", axis=1, inplace=True)
    merged_ingredients.to_csv(os.path.join(os.getcwd(), "../", "cleaned_data", "ingredient_abbrev_transformed.csv"),
                              sep=";", index=False)


def get_grams(ingredient):
    quantities = str(ingredient["qty"]).split()
    units = ingredient["unit"].split() if len(quantities) > 1 else [ingredient["unit"]]

    abbrev_units = list(filter(lambda y: type(y[1]) == str,
                               [(ingredient["GmWt_1"], ingredient["GmWt_Desc1"]),
                                (ingredient["GmWt_2"], ingredient["GmWt_Desc1"])]))

    res, min_distance = 0, float("inf")
    for i in zip(quantities, units):
        for j in abbrev_units:
            cur_distance = jellyfish.levenshtein_distance(i[1], j[1])
            if cur_distance < min_distance:
                min_distance = cur_distance
                res = float(i[0].replace(",", ".")) * j[0]
    return res


if __name__ == "__main__":
    transform_merged_data()
