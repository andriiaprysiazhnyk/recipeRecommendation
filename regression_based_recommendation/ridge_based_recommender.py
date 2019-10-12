import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


def recommend_recipes(recipe, user_id, enc_type):
    preferences = pd.read_csv("preferences_generation/preferences.csv")[str(user_id)]

    if enc_type == "word2vec":
        recipes = pd.read_csv("regression_based_recommendation/word2vec_based_encodings.csv")
    elif enc_type == "mds":
        recipes = pd.read_csv("regression_based_recommendation/mds_encodings.csv")
    elif enc_type == "tfidf":
        recipes = pd.read_csv("regression_based_recommendation/tfidf_encodings.csv")
    else:
        print("incorrect encoding type")
        return

    non_scored_recipes = recipes.loc[preferences[preferences.isna()].index]
    models = pickle.load(open("regression_based_recommendation/word2vec_models.pickle", "rb"))
    model = models[user_id]

    predictions = pd.Series(model.predict(non_scored_recipes), index=non_scored_recipes.index)
    return pd.read_csv("cleaned_data/mapped_recipes.csv", sep=";").loc[predictions.nlargest(3).index]


def generate_models(path, output_name):
    preferences = pd.read_csv("../preferences_generation/preferences.csv")
    recipes = pd.read_csv(path)

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

    pickle.dump(models, open("{}.pickle".format(output_name), "wb"))


if __name__ == "__main__":
    generate_models("word2vec_based_encodings.csv", "word2vec_models")
    generate_models("mds_encodings.csv", "mds_models")
    generate_models("tfidf_encodings.csv", "tfidf_models")
