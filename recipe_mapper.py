import numpy as np
import pandas as pd


def map_recipes():
    ingredients = pd.read_csv("cleaned_data/ingredient_abbrev_transformed.csv", sep=";")
    recipes = pd.read_csv("cleaned_data/recipe.csv", sep=";")
    recipe_ingredient = pd.read_csv("cleaned_data/recipe_ingredient.csv", sep=";")

    recipes = recipes[recipes.apply(lambda x: all_ingredients_present(x, ingredients, recipe_ingredient), axis=1)]
    print(len(recipes))

    add_chemical_columns(recipes, ingredients, recipe_ingredient)
    recipes["ingredients"] = recipes.apply(lambda x: get_names(get_ingredient_ids(x, recipe_ingredient), ingredients),
                                           axis=1)

    non_chemical_columns = ["recipe_id", "title", "author", "url", "tags", "servings", "ingredients"]
    for column in recipes.columns:
        if column not in non_chemical_columns:
            recipes[column] /= recipes["servings"]

    recipes.drop("servings", axis=1, inplace=True)
    recipes["recipe_id"] = np.arange(0, len(recipes))
    recipes.to_csv("cleaned_data/mapped_recipes.csv", sep=";", index=False)


def all_ingredients_present(recipe, all_ingredients, recipe_ingredient):
    ingredient_ids = get_ingredient_ids(recipe, recipe_ingredient)

    for ingredient_id in ingredient_ids:
        if ingredient_id not in list(all_ingredients["ingredient_id"]):
            return False

    return True


def add_chemical_columns(recipes, ingredients, recipe_ingredient):
    non_chemical_columns = ["ingredient_id", "base", "description", "Shrt_Desc"]
    for column in ingredients.columns:
        if column not in non_chemical_columns:
            recipes[column] = recipes.apply(
                lambda x: get_total_amount(get_ingredient_ids(x, recipe_ingredient), ingredients, column), axis=1)


def get_total_amount(ingredient_ids, all_ingredients, column):
    return sum(all_ingredients[all_ingredients["ingredient_id"].apply(lambda x: x in ingredient_ids)][column])


def get_ingredient_ids(recipe, recipe_ingredient):
    return list(recipe_ingredient[recipe_ingredient["recipe_id"] == recipe["recipe_id"]]["ingredient_id"])


def get_names(ingredient_ids, all_ingredients):
    names = list(all_ingredients[all_ingredients["ingredient_id"].apply(lambda x: x in ingredient_ids)]["base"])

    res = ""
    for name in names:
        res += name
        res += "|"

    return res[:-1]


if __name__ == "__main__":
    map_recipes()
