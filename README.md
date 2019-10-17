Data should be available in ./data folder
Pre-trained on Google-News Word2Vec model should be available in ./embedding folder

data_cleaning.data_cleaner.py creates new folder ./cleaned_data with clean data in it
data_merging.merger.py merge ingredients and abbrev (chemicals) tables
data_merging.merged_data_transformer.py transform merged data into new table
recipe_mapper.py map all recipes with its ingredients into one table
recipe_comparator.py encapsulates all logic with recipes similarity metric
preferences_generation.fake_user_generator.py generates fake user "vegetarian" preferences
*_recommendation folders include 2 main modules:
  1) Module that creates a model or prepare data for prediction
  2) Module that recommends recipes based on user preferences and previous module outputs
 
All another modules are self-descriptive
