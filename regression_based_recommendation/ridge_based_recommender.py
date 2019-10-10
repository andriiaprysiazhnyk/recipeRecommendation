import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


PATH_TO_RECIPES = "regression_based_recommendation/word2vec_based_encodings.csv"


def recommend_recipes(recipe, user_id):
    preferences = pd.read_csv("preferences_generation/preferences.csv")[str(user_id)]
    recipes = pd.read_csv(PATH_TO_RECIPES)

    non_scored_recipes = recipes.loc[preferences[preferences.isna()].index]
    models = pickle.load(open("regression_based_recommendation/trained_models.pickle", "rb"))
    model = models[user_id]

    predictions = pd.Series(model.predict(non_scored_recipes), index=non_scored_recipes.index)
    return pd.read_csv("cleaned_data/mapped_recipes.csv", sep=";").loc[predictions.nlargest(3).index]


def generate_models():
    preferences = pd.read_csv("../preferences_generation/preferences.csv")
    recipes = pd.read_csv(PATH_TO_RECIPES)

    models = {}
    for i in range(preferences.shape[1]):
        user_pref = preferences[str(i)]
        user_pref = user_pref[user_pref.notna()]
        scored_recipes = recipes.loc[user_pref.index]

        grid = GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}, cv=4)
        grid.fit(scored_recipes, user_pref)

        ridge = Ridge(**grid.best_params_)
        ridge.fit(scored_recipes, user_pref)
        models[i] = ridge

    pickle.dump(models, open("trained_models.pickle", "wb"))


if __name__ == "__main__":
    generate_models()
