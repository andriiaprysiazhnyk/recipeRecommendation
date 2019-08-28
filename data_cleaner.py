import os

import pandas as pd
from io import StringIO


OUTPUT_DIRECTORY = "cleaned_data"
INPUT_DIRECTORY = "data"


def process_data(output_path):
    for data in read_data():
        drop_empty_columns(data[0])
        data[0].to_csv(os.path.join(output_path, data[1]), sep=";", index=False)


def read_data():
    abbrev = pd.read_csv("{}/abbrev.csv".format(INPUT_DIRECTORY), sep=";")

    incorrect_rows = list(range(2295, 2300))  # manually identified incorrect instances in dataset
    recipes = pd.read_csv("{}/recipe.csv".format(INPUT_DIRECTORY), sep=";", skiprows=incorrect_rows)

    recipe_ingredient = pd.read_csv("{}/recipe_ingredient.csv".format(INPUT_DIRECTORY))
    recipe_ingredient = recipe_ingredient[recipe_ingredient["recipe_id"].apply(lambda x: x not in incorrect_rows)]

    with open("{}/ingredient.csv".format(INPUT_DIRECTORY), "r") as f:
        content = f.read()
        ingredients = pd.read_csv(StringIO(content.replace("; ", ", ")), sep=";")

    file_names = ["abbrev.csv", "recipe.csv", "recipe_ingredient.csv", "ingredient.csv"]
    return zip([abbrev, recipes, recipe_ingredient, ingredients], file_names)


def drop_empty_columns(df):
    for column in df.columns:
        if df[column].value_counts().size < 2:
            df.drop(column, axis=1, inplace=True)


if __name__ == "__main__":
    clean_data_path = os.path.join(os.getcwd(), OUTPUT_DIRECTORY)

    if not os.path.exists(clean_data_path):
        os.makedirs(clean_data_path)
        process_data(clean_data_path)
