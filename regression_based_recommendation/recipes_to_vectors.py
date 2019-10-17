import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from recipe_comparator import seq2vec


def tfidf_vectorizing(chemicals):
    recipes = pd.read_csv("../cleaned_data/mapped_recipes.csv", sep=";")
    ingredients = recipes.ingredients.apply(lambda x: x.replace("|", " "))

    vectorizer = TfidfVectorizer()
    ingredient_encodings = pd.DataFrame(vectorizer.fit_transform(ingredients).toarray())

    pca = PCA(n_components=92)
    reduced_encodings = pd.DataFrame(pca.fit_transform(ingredient_encodings))
    print("Explained variance ratio (TF-IDF):", sum(pca.explained_variance_ratio_))

    res = pd.concat([reduced_encodings, chemicals], axis=1)
    res = (res - res.min()) / (res.max() - res.min())

    res.to_csv("tfidf_encodings.csv", index=False)


def mds_vectorizing():
    distances = 1 - pd.read_csv("../similarity_based_recommendation/recipes_similarities.csv")
    mds = MDS(n_components=100, n_init=10)
    encodings = pd.DataFrame(mds.fit_transform(distances))
    encodings.to_csv("mds_encodings.csv", index=False)


def word2vec_based_vectorizing(chemicals):
    recipes = pd.read_csv("../cleaned_data/mapped_recipes.csv", sep=";")
    ingredients = recipes.ingredients.apply(lambda x: seq2vec(x.replace("|", " ")).reshape(-1))
    ingredients = pd.DataFrame(np.stack(list(ingredients)))

    pca = PCA(n_components=92)
    reduced_encodings = pd.DataFrame(pca.fit_transform(ingredients))
    print("Explained variance ratio (word2vec):", sum(pca.explained_variance_ratio_))

    res = pd.concat([reduced_encodings, chemicals], axis=1)
    res = (res - res.min()) / (res.max() - res.min())

    res.to_csv("word2vec_based_encodings.csv", index=False)


def get_chemicals():
    recipes = pd.read_csv("../cleaned_data/mapped_recipes.csv", sep=";")
    non_chemical_columns = ["recipe_id", "title", "author", "url", "tags", "ingredients"]
    chemicals_columns = list(recipes.columns)
    for non_chemical in non_chemical_columns:
        chemicals_columns.remove(non_chemical)

    chemicals = recipes.drop(non_chemical_columns, axis=1)
    chemicals = (chemicals - chemicals.min()) / (chemicals.max() - chemicals.min())
    pca = PCA(n_components=8)
    reduced_chemicals = pd.DataFrame(pca.fit_transform(chemicals))
    print("Explained variance ratio (chemicals):", sum(pca.explained_variance_ratio_))
    return reduced_chemicals


if __name__ == "__main__":
    chemical_elements = get_chemicals()
    tfidf_vectorizing(chemical_elements)
    mds_vectorizing()
    word2vec_based_vectorizing(chemical_elements)
