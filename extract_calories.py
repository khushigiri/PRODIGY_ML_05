import pandas as pd

food = pd.read_csv("data/usda/food.csv")
nutrient = pd.read_csv("data/usda/nutrient.csv")
food_nutrient = pd.read_csv("data/usda/food_nutrient.csv")

# Get Energy nutrient ID
energy_id = nutrient[nutrient['name'] == 'Energy']['id'].values[0]

# Filter calories
calories = food_nutrient[food_nutrient['nutrient_id'] == energy_id]

# Merge
result = calories.merge(food, on='fdc_id')

print(result[['description', 'amount']].head())