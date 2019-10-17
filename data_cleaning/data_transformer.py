import re
import pandas as pd
from data_cleaning.ingredients_mv_imputer import impute_ingredients_mv
from data_cleaning.chemicals_mv_imputer import impute_abbrev_mv


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
    impute_abbrev_mv(abbrev)
    abbrev = abbrev[abbrev.apply(lambda x: pd.isna(x["GmWt_Desc1"]) or not pd.isna(x["GmWt_1"]), axis=1)]
    transform_units(abbrev, "GmWt_1", "GmWt_Desc1")
    transform_units(abbrev, "GmWt_2", "GmWt_Desc2")
    return abbrev


def transform_units(abbrev, unit_name, unit_desc):
    unit_grams = abbrev[unit_name].apply(lambda x: x if pd.isna(x) else float(x.replace(",", ".")))
    unit_amount = abbrev[unit_desc].apply(lambda x: x if isinstance(x, float) else float(x.split()[0].replace(",", ".")))

    abbrev[unit_name] = unit_grams / unit_amount
    abbrev[unit_desc] = abbrev[unit_desc].apply(lambda x: x if pd.isna(x) else " ".join(x.split()[1:]))

