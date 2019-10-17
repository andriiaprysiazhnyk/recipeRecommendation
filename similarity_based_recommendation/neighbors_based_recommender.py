import pandas as pd


def recommend_recipes(recipe, user_id):
    recipes = pd.read_csv("cleaned_data/mapped_recipes.csv", sep=";")
    preferences = pd.read_csv("preferences_generation/preferences.csv")[str(user_id)]
    similarities = pd.read_csv("similarity_based_recommendation/recipes_similarities.csv")

    scored = preferences[preferences.notna()]
    not_scored = preferences[preferences.isna()]
    not_scored = pd.Series(list(not_scored.index), index=not_scored.index)

    predicted_scores = not_scored.apply(lambda x: predict_score(x, similarities, scored))
    return recipes.loc[predicted_scores.nlargest(3).index]


def predict_score(recipe_id, similarities, scores):
    similarities = similarities[str(recipe_id)][scores.index]
    similarities[recipe_id] = 0
    max_similarities = similarities.nlargest(3)
    res = sum(scores[max_similarities.index] * max_similarities) / 3
    return res
