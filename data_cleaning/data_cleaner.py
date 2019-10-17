import os
import shutil

import pandas as pd
from io import StringIO
from data_cleaning.data_transformer import transform_abbrev, transform_ingredients

OUTPUT_DIRECTORY = "cleaned_data"
INPUT_DIRECTORY = "data"


def process_data(output_path):
    for data in read_data():
        drop_empty_columns(data[0])
        data[0].to_csv(os.path.join(output_path, data[1]), sep=";", index=False)


def read_data():
    input_path = os.path.join(os.getcwd(), "../", INPUT_DIRECTORY)
    abbrev = pd.read_csv(os.path.join(input_path, "abbrev.csv"), sep=";")
    abbrev = transform_abbrev(abbrev)

    incorrect_rows = list(range(2295, 2300))  # manually identified incorrect instances in dataset
    recipes = pd.read_csv(os.path.join(input_path, "recipe.csv"), sep=";", skiprows=incorrect_rows)

    recipe_ingredient = pd.read_csv(os.path.join(input_path, "recipe_ingredient.csv"))
    recipe_ingredient = recipe_ingredient[recipe_ingredient["recipe_id"].apply(lambda x: x not in incorrect_rows)]

    with open(os.path.join(input_path, "ingredient.csv"), "r") as f:
        content = f.read()
        ingredients = pd.read_csv(StringIO(content.replace("; ", "~")), sep=";")
        ingredients = ingredients.applymap(lambda x: x.replace("~", "; ") if isinstance(x, str) else x)
        ingredients = transform_ingredients(ingredients)

    file_names = ["abbrev.csv", "recipe.csv", "recipe_ingredient.csv", "ingredient.csv"]
    return zip([abbrev, recipes, recipe_ingredient, ingredients], file_names)


def drop_empty_columns(df):
    for column in df.columns:
        if df[column].value_counts().size < 2:
            df.drop(column, axis=1, inplace=True)


if __name__ == "__main__":
    clean_data_path = os.path.join(os.getcwd(), "../", OUTPUT_DIRECTORY)

    if os.path.exists(clean_data_path):
        shutil.rmtree(clean_data_path)

    os.makedirs(clean_data_path)
    process_data(clean_data_path)

