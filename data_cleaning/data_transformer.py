import re
import pandas as pd
from data_cleaning.ingredients_mv_imputer import impute_ingredients_mv


def transform_ingredients(ingredients):
    ingredients.drop(columns="ndb", axis=1, inplace=True)
    ingredients.dropna(subset=["base"], inplace=True)
    transform_base(ingredients)
    ingredients = ingredients[ingredients.qty.apply(check_qty)]
    ingredients = drop_inconsistent(ingredients)
    impute_ingredients_mv(ingredients)
    return ingredients


def transform_base(ingredients):
    ingredients["base"] = ingredients["base"] \
        .apply(lambda x: re.sub("[^ ;""a-zA-Z]", " ", x.lower())) \
        .apply(lambda x: re.sub("\s+", " ", x)) \
        .apply(lambda x: x.replace(" plus", ";"))


def drop_inconsistent(ingredients):
    return ingredients[ingredients.apply(
        lambda x: pd.isna(x["qty"]) or pd.isna(x["unit"]) or len(x["unit"].split()) == len(
            x["qty"].split()) or len(x["qty"].split()) == 1, axis=1)]


def check_qty(s):
    if pd.isna(s):
        return True

    letters = "0123456789., "
    for letter in s:
        if letter not in letters:
            return False
    return True


def transform_abbrev(abbrev):
    abbrev.dropna(subset=["GmWt_Desc1", "GmWt_Desc2"], how="all", inplace=True)
    abbrev["Shrt_Desc"] = abbrev["Shrt_Desc"].apply(lambda x: x.lower())
