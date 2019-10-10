import numpy as np
import pandas as pd
import random


def generate_preferences():
    recipes = pd.read_csv("../cleaned_data/mapped_recipes.csv", sep=";")
    recipes_number, users_number = len(recipes), 100

    preferences = np.empty((recipes_number, users_number))
    preferences[:] = float("NaN")

    for i in range(users_number):
        evaluation_prob = random.uniform(0, 1)
        satisfactory_prob = random.uniform(0.2, 0.8)

        for j in range(recipes_number):
            if np.random.choice([0, 1], p=[evaluation_prob, 1 - evaluation_prob]) == 0:
                if np.random.choice([0, 1], p=[satisfactory_prob, 1 - satisfactory_prob]) == 0:
                    preferences[j, i] = random.uniform(5, 10)
                else:
                    preferences[j, i] = random.uniform(0, 5)

    pd.DataFrame(preferences).to_csv("preferences.csv", index=False)


if __name__ == "__main__":
    generate_preferences()
